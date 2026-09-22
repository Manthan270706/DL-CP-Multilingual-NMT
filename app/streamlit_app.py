"""Multilingual Neural Machine Translation Web Application.
Master Decision Register Section 10, PRD Section 24-26, FSD Section 9, Runbook Section 3 (TASK-021).
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.inference.engine import TranslationInferenceEngine
from src.registry.language_registry import LanguageRegistry
from src.registry.model_registry import ModelRegistry

# Page Config
st.set_page_config(
    page_title="Multilingual Neural Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        min-height: 120px;
        font-size: 1.15rem;
        color: #0F172A;
    }
    .metric-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Translation Inference Engine...")
def get_inference_engine() -> TranslationInferenceEngine:
    """Instantiate and cache the global inference engine."""
    return TranslationInferenceEngine()


def main():
    engine = get_inference_engine()

    # Title & Header
    st.markdown(
        '<div class="main-header">🌐 Multilingual Neural Machine Translation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Transformer-based Deep Learning Translation Platform '
        "(<b>MarianMT</b>, <b>mBART-50</b>, <b>NLLB-200</b>)</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Translation Configuration")

        # Source Language
        src_lang = st.selectbox(
            "Source Language",
            options=["en"],
            format_func=lambda x: f"{LanguageRegistry.get_language(x).name} ({x})",
            index=0,
            disabled=True,
            help="English is the primary source language.",
        )

        # Target Language
        supported_targets = LanguageRegistry.get_supported_targets()
        tgt_lang = st.selectbox(
            "Target Language",
            options=supported_targets,
            format_func=lambda x: f"{LanguageRegistry.get_language(x).name} ({x}) - {LanguageRegistry.get_language(x).resource_level.title()}",
            index=0,
        )

        # Model Selector
        model_options = [
            ("nllb200", "NLLB-200 (Primary Multilingual)"),
            ("mbart50", "mBART-50 (Many-to-Many Multilingual)"),
            ("marian", "MarianMT (Pair-Specific Baseline)"),
        ]
        selected_model = st.selectbox(
            "Model Architecture",
            options=[opt[0] for opt in model_options],
            format_func=lambda x: dict(model_options)[x],
            index=0,
            help="Select the Transformer model to generate the translation.",
        )

        # Generation Hyperparameters
        with st.expander("🛠️ Advanced Generation Parameters", expanded=False):
            num_beams = st.slider(
                "Beam Width (num_beams)", min_value=1, max_value=8, value=5, step=1
            )
            max_new_tokens = st.slider(
                "Max New Tokens", min_value=32, max_value=256, value=128, step=16
            )

        # Comparison Mode Toggle
        comparison_mode = st.checkbox("🔍 Model Comparison Mode (Run all 3 models)", value=False)

        st.markdown("---")
        st.caption("Hardware: **NVIDIA RTX 5050 Laptop GPU (8 GB VRAM)**")
        st.caption("Deep Learning: **PyTorch + Hugging Face Transformers**")

    # Example Sentences Selector
    examples = {
        "General AI": "Artificial intelligence is transforming the world.",
        "Deep Learning": "Deep learning models require computational power.",
        "Translation": "Machine translation helps bridge language barriers.",
        "Science / Nature": "The sun rises in the east and sets in the west.",
        "Education": "Education is the key to personal and social development.",
        "Custom Input": "",
    }

    selected_example_key = st.selectbox("📝 Quick Examples", options=list(examples.keys()), index=0)
    default_text = examples[selected_example_key]

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 Source Text (English)")
        input_text = st.text_area(
            label="Enter text in English",
            value=default_text,
            height=160,
            placeholder="Type or paste your English text here...",
            label_visibility="collapsed",
        )

        char_count = len(input_text)
        word_count = len(input_text.split()) if input_text else 0
        st.caption(f"Characters: {char_count} | Words: {word_count}")

        translate_clicked = st.button("🚀 Translate", type="primary", use_container_width=True)

    with col2:
        target_name = LanguageRegistry.get_language(tgt_lang).name
        st.markdown(f"### 📤 Translation ({target_name})")

        if translate_clicked or input_text:
            if not input_text.strip():
                st.warning("Please enter some text to translate.")
            else:
                if comparison_mode:
                    # Run comparison across all 3 models
                    with st.spinner(
                        "Generating translations with MarianMT, mBART-50, and NLLB-200..."
                    ):
                        models_to_run = ["marian", "mbart50", "nllb200"]
                        for m_id in models_to_run:
                            try:
                                res = engine.translate_single(
                                    text=input_text,
                                    src_lang=src_lang,
                                    tgt_lang=tgt_lang,
                                    model_id=m_id,
                                    num_beams=num_beams,
                                    max_new_tokens=max_new_tokens,
                                )
                                m_name = ModelRegistry.get_model(res["model_id"]).display_name
                                st.markdown(f"**{m_name}**")
                                st.markdown(
                                    f'<div class="result-box">{res["translated_text"]}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<span class="metric-badge">⚡ Latency: {res["latency_ms"]} ms</span>',
                                    unsafe_allow_html=True,
                                )
                                st.write("")
                            except Exception as e:
                                st.error(f"Error generating translation with {m_id}: {e}")
                else:
                    # Single model translation
                    with st.spinner(f"Translating using {dict(model_options)[selected_model]}..."):
                        try:
                            res = engine.translate_single(
                                text=input_text,
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                model_id=selected_model,
                                num_beams=num_beams,
                                max_new_tokens=max_new_tokens,
                            )
                            st.markdown(
                                f'<div class="result-box">{res["translated_text"]}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<span class="metric-badge">⚡ Latency: {res["latency_ms"]} ms</span>'
                                f'<span class="metric-badge">🤖 Model: {res["model_id"]}</span>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error(f"Translation Error: {e}")
        else:
            st.markdown(
                '<div class="result-box" style="color: #94A3B8;">Translated text will appear here...</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
