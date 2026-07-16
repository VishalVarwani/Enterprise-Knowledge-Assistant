"""
pii_filter.py
-------------
PII detection and redaction using Microsoft Presidio.

Why Presidio:
  - Microsoft-maintained, production-deployed at scale (Azure Cognitive Services)
  - 50+ built-in entity recognizers: EMAIL_ADDRESS, PHONE_NUMBER, PERSON,
    CREDIT_CARD, US_SSN, IP_ADDRESS, IBAN_CODE, NRP, LOCATION, DATE_TIME...
  - Extensible: add custom recognizers (employee IDs, internal project codes)
  - Replaceable operators: can redact, replace, encrypt, or hash PII
  - Both detection (analyzer) and anonymization (anonymizer) in one library

Two use cases here:
  1. INPUT PII: user queries containing personal data (email, SSN, etc.)
     → Detect and warn user to rephrase without PII
     → Optionally redact before logging (GDPR compliance)

  2. OUTPUT PII: KB documents containing sensitive records that shouldn't
     be returned verbatim in API responses
     → Redact in the generated answer before returning to user

Entity types we care about for enterprise KB:
  - PERSON: employee names in documents
  - EMAIL_ADDRESS: contact emails
  - PHONE_NUMBER: support lines
  - CREDIT_CARD: financial docs
  - US_SSN, EU_TAX_ID: HR/payroll docs
  - IBAN_CODE: finance docs
  - IP_ADDRESS: IT/security docs
  - LOCATION: for privacy (not typically redacted in enterprise KB)

We use a threshold of 0.7 confidence; below this is likely a false positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    raise ImportError(
        "presidio not installed. Run: pip install presidio-analyzer presidio-anonymizer"
    )

# Entity types to check in user queries (input PII)
# Don't flag LOCATION (too many false positives for enterprise KB queries)
INPUT_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "IP_ADDRESS",
    "DATE_TIME",  # DOB in queries is a red flag
]

# Entity types to redact in LLM output (output PII)
# More permissive: allow email/phone in KB answers (they're contact info)
OUTPUT_PII_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "CRYPTO",
    "MEDICAL_LICENSE",
]

# Minimum confidence score to flag as PII (0.0–1.0)
# 0.7 balances false positive rate vs. missed detections
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class PIIDetectionResult:
    """Result of a PII analysis pass."""
    has_pii: bool
    entities_found: list[dict]     # [{"type": "EMAIL_ADDRESS", "score": 0.95, "start": 5, "end": 25}]
    redacted_text: Optional[str]   # Text with PII replaced by placeholders
    entity_types: list[str]        # Unique entity type names found
    should_block: bool = False     # True for high-severity PII (SSN, credit cards)

    def severity(self) -> str:
        """
        Classify severity based on entity types found.
        Used to decide whether to block (high) or warn (low/medium).
        """
        high_risk = {"US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER"}
        medium_risk = {"EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"}

        found = set(self.entity_types)
        if found & high_risk:
            return "high"
        elif found & medium_risk:
            return "medium"
        return "low"


class PIIFilter:
    """
    Detects and optionally redacts PII in text.

    Singleton pattern: AnalyzerEngine is expensive to initialize
    (loads NLP models). Initialize once, reuse across requests.
    """

    _analyzer: Optional[AnalyzerEngine] = None
    _anonymizer: Optional[AnonymizerEngine] = None

    def __init__(self):
        if PIIFilter._analyzer is None:
            logger.info("Initializing Presidio analyzer...")
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer import PatternRecognizer, Pattern

            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

            # Add explicit regex-based SSN recognizer (en_core_web_sm misses these)
            ssn_recognizer = PatternRecognizer(
                supported_entity="US_SSN",
                patterns=[
                    Pattern(
                        name="ssn_dashes",
                        regex=r"\b\d{3}-\d{2}-\d{4}\b",
                        score=0.85,
                    ),
                    Pattern(
                        name="ssn_spaces",
                        regex=r"\b\d{3}\s\d{2}\s\d{4}\b",
                        score=0.75,
                    ),
                ],
            )
            analyzer.registry.add_recognizer(ssn_recognizer)

            PIIFilter._analyzer = analyzer
            PIIFilter._anonymizer = AnonymizerEngine()
            logger.info("Presidio initialized (en_core_web_sm + custom SSN recognizer).")

        self.analyzer = PIIFilter._analyzer
        self.anonymizer = PIIFilter._anonymizer

    def check_input(self, text: str) -> PIIDetectionResult:
        """Alias for analyze_input (used by tests and pipeline)."""
        return self.analyze_input(text)

    def check_output(self, text: str) -> PIIDetectionResult:
        """Alias for analyze_output (used by tests and pipeline)."""
        return self.analyze_output(text)

    def analyze_input(self, text: str) -> PIIDetectionResult:
        """
        Analyze user query input for PII.

        Used to:
          - Block queries containing high-risk PII (SSN, credit cards)
          - Warn on medium-risk PII (email, phone)
          - Redact before logging for GDPR compliance
        """
        return self._analyze(text, entities=INPUT_PII_ENTITIES)

    def analyze_output(self, text: str) -> PIIDetectionResult:
        """
        Analyze LLM-generated output for PII that shouldn't be returned.

        Used to scrub financial identifiers from generated answers.
        Less aggressive than input analysis (KB docs legitimately contain emails, etc.)
        """
        return self._analyze(text, entities=OUTPUT_PII_ENTITIES)

    def redact(self, text: str, entities: list[str] = INPUT_PII_ENTITIES) -> str:
        """
        Return text with PII replaced by type placeholders.

        Example: "Contact john.doe@company.com" → "Contact <EMAIL_ADDRESS>"

        The placeholder format makes it clear what was removed without
        hiding the fact that information existed.
        """
        analyzer_results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=entities,
            score_threshold=CONFIDENCE_THRESHOLD,
        )

        if not analyzer_results:
            return text

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators={
                # Replace each entity type with its type name as a placeholder
                entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                for entity in entities
            },
        )
        return anonymized.text

    def _analyze(self, text: str, entities: list[str]) -> PIIDetectionResult:
        """Core analysis logic shared by input and output analyzers."""
        if not text or not text.strip():
            return PIIDetectionResult(
                has_pii=False, entities_found=[], redacted_text=text, entity_types=[], should_block=False
            )

        results: list[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language="en",
            entities=entities,
            score_threshold=CONFIDENCE_THRESHOLD,
        )

        if not results:
            return PIIDetectionResult(
                has_pii=False, entities_found=[], redacted_text=text, entity_types=[], should_block=False
            )

        entities_found = [
            {
                "type": r.entity_type,
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
                "value": text[r.start:r.end],  # The matched text segment
            }
            for r in results
        ]

        entity_types = list({r.entity_type for r in results})

        # Always produce a redacted version for logging
        redacted = self.redact(text, entities)

        high_risk = {"US_SSN", "CREDIT_CARD", "IBAN_CODE", "US_BANK_NUMBER"}
        found_high_risk = bool(set(entity_types) & high_risk)

        return PIIDetectionResult(
            has_pii=True,
            entities_found=entities_found,
            redacted_text=redacted,
            entity_types=entity_types,
            should_block=found_high_risk,
        )
