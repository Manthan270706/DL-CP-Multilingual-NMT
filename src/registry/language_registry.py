"""Authoritative Language Registry for Multilingual NMT.
Master Decision Register Section 2 & Manifest Section 2.
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class LanguageInfo:
    code: str  # ISO 639-1 (e.g., 'en', 'hi', 'mr', 'fr', 'de')
    name: str  # Full name (e.g., 'English', 'Hindi')
    script: str  # Script code (e.g., 'Latn', 'Deva')
    nllb_code: str  # NLLB-200 code (e.g., 'eng_Latn', 'hin_Deva')
    mbart_code: str  # mBART-50 code (e.g., 'en_XX', 'hi_IN')
    resource_level: str  # 'high', 'medium-high', 'lower-resource'


class LanguageRegistry:
    """Authoritative language metadata registry."""

    _PRIMARY_SOURCE: str = "en"
    _SUPPORTED_TARGETS: Set[str] = {"hi", "mr", "fr", "de"}
    _SUPPORTED_DIRECTIONS: Set[str] = {"en-hi", "en-mr", "en-fr", "en-de"}

    _LANGUAGES: Dict[str, LanguageInfo] = {
        "en": LanguageInfo(
            code="en",
            name="English",
            script="Latn",
            nllb_code="eng_Latn",
            mbart_code="en_XX",
            resource_level="high",
        ),
        "hi": LanguageInfo(
            code="hi",
            name="Hindi",
            script="Deva",
            nllb_code="hin_Deva",
            mbart_code="hi_IN",
            resource_level="medium-high",
        ),
        "mr": LanguageInfo(
            code="mr",
            name="Marathi",
            script="Deva",
            nllb_code="mar_Deva",
            mbart_code="mr_IN",
            resource_level="lower-resource",
        ),
        "fr": LanguageInfo(
            code="fr",
            name="French",
            script="Latn",
            nllb_code="fra_Latn",
            mbart_code="fr_XX",
            resource_level="high",
        ),
        "de": LanguageInfo(
            code="de",
            name="German",
            script="Latn",
            nllb_code="deu_Latn",
            mbart_code="de_DE",
            resource_level="high",
        ),
    }

    @classmethod
    def get_language(cls, code: str) -> LanguageInfo:
        """Retrieve language metadata by ISO code."""
        code = code.lower().strip()
        if code not in cls._LANGUAGES:
            raise ValueError(
                f"Unsupported language code '{code}'. Supported codes: {list(cls._LANGUAGES.keys())}"
            )
        return cls._LANGUAGES[code]

    @classmethod
    def get_all_languages(cls) -> List[LanguageInfo]:
        """Return all supported language infos."""
        return list(cls._LANGUAGES.values())

    @classmethod
    def get_primary_source(cls) -> str:
        """Return the primary source language code ('en')."""
        return cls._PRIMARY_SOURCE

    @classmethod
    def get_supported_targets(cls) -> List[str]:
        """Return list of supported target language codes."""
        return sorted(list(cls._SUPPORTED_TARGETS))

    @classmethod
    def get_supported_directions(cls) -> List[str]:
        """Return list of supported translation directions (e.g. ['en-de', 'en-fr', 'en-hi', 'en-mr'])."""
        return sorted(list(cls._SUPPORTED_DIRECTIONS))

    @classmethod
    def is_direction_supported(cls, src: str, tgt: str) -> bool:
        """Check if a specific src->tgt translation direction is supported."""
        direction = f"{src.lower().strip()}-{tgt.lower().strip()}"
        return direction in cls._SUPPORTED_DIRECTIONS

    @classmethod
    def validate_direction(cls, src: str, tgt: str) -> None:
        """Validate src->tgt direction or raise ValueError."""
        src = src.lower().strip()
        tgt = tgt.lower().strip()
        if not cls.is_direction_supported(src, tgt):
            raise ValueError(
                f"Translation direction '{src}->{tgt}' is not in the frozen primary scope. "
                f"Supported directions: {cls.get_supported_directions()}"
            )

    @classmethod
    def get_nllb_code(cls, lang_code: str) -> str:
        """Get NLLB-200 code for a given ISO language code."""
        return cls.get_language(lang_code).nllb_code

    @classmethod
    def get_mbart_code(cls, lang_code: str) -> str:
        """Get mBART-50 code for a given ISO language code."""
        return cls.get_language(lang_code).mbart_code
