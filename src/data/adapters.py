"""Dataset Adapters for Multilingual NMT.
Master Decision Register Section 5 & Spec Section 2 (TASK-004).
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.registry.language_registry import LanguageRegistry


class DatasetAdapter:
    """Adapts various raw parallel corpus formats to the canonical schema:
    [sample_id, source_language, target_language, source_text, target_text, dataset_source]
    """

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
        src_col: str,
        tgt_col: str,
        source_name: str,
    ) -> pd.DataFrame:
        """Convert any generic DataFrame into canonical parallel schema."""
        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        canonical = pd.DataFrame()
        canonical["source_text"] = df[src_col].astype(str)
        canonical["target_text"] = df[tgt_col].astype(str)
        canonical["source_language"] = src_lang.lower().strip()
        canonical["target_language"] = tgt_lang.lower().strip()
        canonical["dataset_source"] = source_name
        canonical["sample_id"] = [
            f"{src_lang}_{tgt_lang}_{source_name}_{i:07d}" for i in range(len(canonical))
        ]
        return canonical[
            [
                "sample_id",
                "source_language",
                "target_language",
                "source_text",
                "target_text",
                "dataset_source",
            ]
        ]

    @staticmethod
    def load_from_hf_dataset(
        dataset_name: str,
        src_lang: str,
        tgt_lang: str,
        subset: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load parallel pairs from a Hugging Face dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Hugging Face `datasets` package is required to load remote datasets."
            )

        LanguageRegistry.validate_direction(src_lang, tgt_lang)

        # Standard Hugging Face translation loading patterns
        if subset:
            ds = load_dataset(dataset_name, subset, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        rows = []
        # Check standard format 1: 'translation' dict column containing {src: text, tgt: text}
        if "translation" in ds.column_names:
            for item in ds["translation"]:
                if src_lang in item and tgt_lang in item:
                    rows.append(
                        {
                            "source_text": item[src_lang],
                            "target_text": item[tgt_lang],
                        }
                    )
        # Check format 2: separate columns
        elif src_lang in ds.column_names and tgt_lang in ds.column_names:
            for item in ds:
                rows.append(
                    {
                        "source_text": item[src_lang],
                        "target_text": item[tgt_lang],
                    }
                )
        else:
            raise ValueError(
                f"Cannot find columns or translation dict for {src_lang}->{tgt_lang} in {dataset_name} columns: {ds.column_names}"
            )

        df = pd.DataFrame(rows)
        source_name = f"{dataset_name}_{subset or 'default'}"
        return DatasetAdapter.from_dataframe(
            df=df,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_col="source_text",
            tgt_col="target_text",
            source_name=source_name,
        )

    @staticmethod
    def load_iit_bombay_corpus(max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Load the IIT Bombay English-Hindi Parallel Corpus from Hugging Face.

        Per Master Decision Register Section 5, this is the authoritative primary
        source for EN→HI. Uses the ``cfilt/iitb-english-hindi`` dataset on HF Hub.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Hugging Face `datasets` package required.")

        ds = load_dataset("cfilt/iitb-english-hindi", split="train")
        orig_len = len(ds)
        if max_samples and max_samples < orig_len:
            ds = ds.select(range(max_samples))

        rows = []
        for item in ds:
            translation = item.get("translation", {})
            en_text = translation.get("en", "").strip()
            hi_text = translation.get("hi", "").strip()
            if en_text and hi_text:
                rows.append({"source_text": en_text, "target_text": hi_text})

        df = pd.DataFrame(rows)
        canonical = DatasetAdapter.from_dataframe(
            df=df,
            src_lang="en",
            tgt_lang="hi",
            src_col="source_text",
            tgt_col="target_text",
            source_name="iitb_english_hindi",
        )
        metadata = {
            "dataset_name": "cfilt/iitb-english-hindi",
            "configuration": "default",
            "provider": "Center for Indian Language Technology (CFILT), IIT Bombay",
            "license": "CC-BY-NC-SA 4.0",
            "original_size": orig_len,
            "version": "1.0.0",
            "split_loaded": "train",
        }
        return canonical, metadata

    @staticmethod
    def load_opus_dataset(
        src_lang: str,
        tgt_lang: str,
        opus_name: str = "opus_books",
        max_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Load an OPUS parallel corpus from HuggingFace with metadata."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Hugging Face `datasets` package required.")

        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        lang_pair = f"{src_lang}-{tgt_lang}"

        # Determine valid config name (e.g., opus_books uses 'de-en' for German)
        actual_config = lang_pair
        try:
            ds = load_dataset(opus_name, actual_config, split="train")
        except Exception:
            actual_config = f"{tgt_lang}-{src_lang}"
            ds = load_dataset(opus_name, actual_config, split="train")

        orig_len = len(ds)
        if max_samples and max_samples < orig_len:
            ds = ds.select(range(max_samples))

        rows = []
        for item in ds:
            translation = item.get("translation", {})
            src_text = translation.get(src_lang, "").strip()
            tgt_text = translation.get(tgt_lang, "").strip()
            if src_text and tgt_text:
                rows.append({"source_text": src_text, "target_text": tgt_text})

        if not rows:
            raise ValueError(
                f"No translation pairs found in {opus_name} [{actual_config}] for {src_lang}-{tgt_lang}."
            )

        df = pd.DataFrame(rows)
        canonical = DatasetAdapter.from_dataframe(
            df=df,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_col="source_text",
            tgt_col="target_text",
            source_name=f"{opus_name.replace('/', '_')}_{actual_config}",
        )
        metadata = {
            "dataset_name": opus_name,
            "configuration": actual_config,
            "provider": "Language Technology Research Group, University of Helsinki / OPUS",
            "license": "CC-BY-4.0 / Open Data",
            "original_size": orig_len,
            "version": "1.0.0",
            "split_loaded": "train",
        }
        return canonical, metadata

    @staticmethod
    def load_for_direction(
        src_lang: str,
        tgt_lang: str,
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load canonical dataset DataFrame for a language pair."""
        df, _ = DatasetAdapter.load_for_direction_with_metadata(src_lang, tgt_lang, max_samples)
        return df

    @staticmethod
    def load_for_direction_with_metadata(
        src_lang: str,
        tgt_lang: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Auto-select the canonical production dataset with metadata per Master Decision Register.

        Routing table:
          - EN→HI : cfilt/iitb-english-hindi
          - EN→MR : Helsinki-NLP/opus-100 (en-mr)
          - EN→FR : opus_books (en-fr)
          - EN→DE : opus_books (de-en)
        """
        LanguageRegistry.validate_direction(src_lang, tgt_lang)
        direction = f"{src_lang}-{tgt_lang}"

        if direction == "en-hi":
            return DatasetAdapter.load_iit_bombay_corpus(max_samples=max_samples)
        elif direction == "en-mr":
            return DatasetAdapter.load_opus_dataset(
                src_lang="en",
                tgt_lang="mr",
                opus_name="Helsinki-NLP/opus-100",
                max_samples=max_samples,
            )
        elif direction == "en-fr":
            return DatasetAdapter.load_opus_dataset(
                src_lang="en",
                tgt_lang="fr",
                opus_name="opus_books",
                max_samples=max_samples,
            )
        elif direction == "en-de":
            return DatasetAdapter.load_opus_dataset(
                src_lang="en",
                tgt_lang="de",
                opus_name="opus_books",
                max_samples=max_samples,
            )
        else:
            raise ValueError(f"No canonical dataset configured for {direction}")


    @staticmethod
    def create_synthetic_verification_corpus(
        src_lang: str,
        tgt_lang: str,
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """Create a reproducible verified synthetic parallel corpus for offline unit testing & pipeline verification."""
        LanguageRegistry.validate_direction(src_lang, tgt_lang)

        # Verified parallel sentences for all 4 pairs
        parallel_samples = {
            "en-hi": [
                (
                    "Artificial intelligence is transforming the world.",
                    "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",
                ),
                (
                    "Deep learning models require computational power.",
                    "डीप लर्निंग मॉडल के लिए कंप्यूटेशनल शक्ति की आवश्यकता होती है।",
                ),
                (
                    "Machine translation helps bridge language barriers.",
                    "मशीन अनुवाद भाषा की बाधाओं को पाटने में मदद करता है।",
                ),
                (
                    "The sun rises in the east and sets in the west.",
                    "सूर्य पूर्व में उगता है और पश्चिम में अस्त होता है।",
                ),
                (
                    "Education is the key to personal and social development.",
                    "शिक्षा व्यक्तिगत और सामाजिक विकास की कुंजी है।",
                ),
                (
                    "Technology makes communication faster and more accessible.",
                    "प्रौद्योगिकी संचार को तेज और अधिक सुलभ बनाती है।",
                ),
                (
                    "Researchers are developing new neural network architectures.",
                    "शोधकर्ता नए न्यूरल नेटवर्क आर्किटेक्चर विकसित कर रहे हैं।",
                ),
                (
                    "Data preprocessing is an essential step in machine learning.",
                    "डेटा प्रीप्रोसेसिंग मशीन लर्निंग में एक आवश्यक कदम है।",
                ),
                (
                    "Knowledge is power and wisdom is understanding.",
                    "ज्ञान शक्ति है और बुद्धिमत्ता समझ है।",
                ),
                (
                    "We must protect natural resources for future generations.",
                    "हमें भावी पीढ़ियों के लिए प्राकृतिक संसाधनों की रक्षा करनी चाहिए।",
                ),
            ],
            "en-mr": [
                (
                    "Artificial intelligence is transforming the world.",
                    "कृत्रिम बुद्धिमत्ता जग बदलत आहे.",
                ),
                (
                    "Deep learning models require computational power.",
                    "डीप लर्निंग मॉडेल्ससाठी संगणकीय शक्ती आवश्यक आहे.",
                ),
                (
                    "Machine translation helps bridge language barriers.",
                    "मशीन भाषांतर भाषेतील अडथळे दूर करण्यास मदत करते.",
                ),
                (
                    "The sun rises in the east and sets in the west.",
                    "सूर्य पूर्वेला उगवतो आणि पश्चिमेला मावळतो.",
                ),
                (
                    "Education is the key to personal and social development.",
                    "शिक्षण ही वैयक्तिक आणि सामाजिक विकासाची गुरुकिल्ली आहे.",
                ),
                (
                    "Technology makes communication faster and more accessible.",
                    "तंत्रज्ञानामुळे संवाद जलद आणि अधिक सुलभ होतो.",
                ),
                (
                    "Researchers are developing new neural network architectures.",
                    "संशोधक नवीन न्यूरल नेटवर्क आर्किटेक्चर विकसित करत आहेत.",
                ),
                (
                    "Data preprocessing is an essential step in machine learning.",
                    "डेटा प्रीप्रोसेसिंग ही मशीन लर्निंगमधील एक महत्त्वाची पायरी आहे.",
                ),
                (
                    "Knowledge is power and wisdom is understanding.",
                    "ज्ञान ही शक्ती आहे आणि बुद्धिमत्ता म्हणजे समज.",
                ),
                (
                    "We must protect natural resources for future generations.",
                    "आपण भावी पिढ्यांसाठी नैसर्गिक संसाधनांचे रक्षण केले पाहिजे.",
                ),
            ],
            "en-fr": [
                (
                    "Artificial intelligence is transforming the world.",
                    "L'intelligence artificielle transforme le monde.",
                ),
                (
                    "Deep learning models require computational power.",
                    "Les modèles d'apprentissage profond nécessitent de la puissance de calcul.",
                ),
                (
                    "Machine translation helps bridge language barriers.",
                    "La traduction automatique aide à surmonter les barrières linguistiques.",
                ),
                (
                    "The sun rises in the east and sets in the west.",
                    "Le soleil se lève à l'est et se couche à l'ouest.",
                ),
                (
                    "Education is the key to personal and social development.",
                    "L'éducation est la clé du développement personnel et social.",
                ),
                (
                    "Technology makes communication faster and more accessible.",
                    "La technologie rend la communication plus rapide et plus accessible.",
                ),
                (
                    "Researchers are developing new neural network architectures.",
                    "Les chercheurs développent de nouvelles architectures de réseaux neuronaux.",
                ),
                (
                    "Data preprocessing is an essential step in machine learning.",
                    "Le prétraitement des données est une étape essentielle de l'apprentissage automatique.",
                ),
                (
                    "Knowledge is power and wisdom is understanding.",
                    "Le savoir est une puissance et la sagesse est la compréhension.",
                ),
                (
                    "We must protect natural resources for future generations.",
                    "Nous devons protéger les ressources naturelles pour les générations futures.",
                ),
            ],
            "en-de": [
                (
                    "Artificial intelligence is transforming the world.",
                    "Künstliche Intelligenz verändert die Welt.",
                ),
                (
                    "Deep learning models require computational power.",
                    "Deep-Learning-Modelle erfordern Rechenleistung.",
                ),
                (
                    "Machine translation helps bridge language barriers.",
                    "Maschinelle Übersetzung hilft Sprachbarrieren zu überwinden.",
                ),
                (
                    "The sun rises in the east and sets in the west.",
                    "Die Sonne geht im Osten auf und im Westen unter.",
                ),
                (
                    "Education is the key to personal and social development.",
                    "Bildung ist der Schlüssel zur persönlichen und gesellschaftlichen Entwicklung.",
                ),
                (
                    "Technology makes communication faster and more accessible.",
                    "Technologie macht die Kommunikation schneller und zugänglicher.",
                ),
                (
                    "Researchers are developing new neural network architectures.",
                    "Forscher entwickeln neue Architekturen für neuronale Netze.",
                ),
                (
                    "Data preprocessing is an essential step in machine learning.",
                    "Die Datenvorverarbeitung ist ein wesentlicher Schritt beim maschinellen Lernen.",
                ),
                (
                    "Knowledge is power and wisdom is understanding.",
                    "Wissen ist Macht und Weisheit ist Verstehen.",
                ),
                (
                    "We must protect natural resources for future generations.",
                    "Wir müssen die natürlichen Ressourcen für künftige Generationen schützen.",
                ),
            ],
        }

        direction = f"{src_lang}-{tgt_lang}"
        base_pairs = parallel_samples.get(direction, [])
        if not base_pairs:
            raise ValueError(f"No sample templates for direction {direction}")

        # Replicate to target count with indexed variations for testing
        rows = []
        for i in range(n_samples):
            src_template, tgt_template = base_pairs[i % len(base_pairs)]
            if i >= len(base_pairs):
                src_text = f"{src_template} (Sample {i})"
                tgt_text = (
                    f"{tgt_template} (नमुना {i})"
                    if tgt_lang in ["hi", "mr"]
                    else f"{tgt_template} (Exemple {i})"
                )
            else:
                src_text = src_template
                tgt_text = tgt_template
            rows.append(
                {
                    "source_text": src_text,
                    "target_text": tgt_text,
                }
            )

        df = pd.DataFrame(rows)
        return DatasetAdapter.from_dataframe(
            df=df,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_col="source_text",
            tgt_col="target_text",
            source_name="synthetic_verified",
        )
