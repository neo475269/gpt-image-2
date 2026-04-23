import base64
import io
from datetime import datetime

import streamlit as st
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

client = AzureOpenAI(
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-04-01-preview",
)
DEPLOYMENT = st.secrets["AZURE_OPENAI_DEPLOYMENT"]

# ---------------------------------------------------------------------------
# Size / quality helpers
# ---------------------------------------------------------------------------

POPULAR_SIZES = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
]

QUALITY_OPTIONS = ["auto", "low", "medium", "high"]


def _validate_custom_size(w: int, h: int) -> str | None:
    """Return an error message if the custom size violates gpt-image-2 rules."""
    if w <= 0 or h <= 0:
        return "Width and height must be positive."
    if max(w, h) > 3840:
        return "Maximum edge length must be ≤ 3840 px."
    if w % 16 != 0 or h % 16 != 0:
        return "Both edges must be multiples of 16 px."
    long, short = max(w, h), min(w, h)
    if long / short > 3:
        return "Long-to-short edge ratio must not exceed 3:1."
    total = w * h
    if total < 655_360:
        return f"Total pixels ({total:,}) is below the minimum of 655,360."
    if total > 8_294_400:
        return f"Total pixels ({total:,}) exceeds the maximum of 8,294,400."
    return None


def _resolve_size(choice: str, custom_w: int, custom_h: int):
    """Return the size string and any validation error."""
    if choice != "custom":
        return choice, None
    err = _validate_custom_size(custom_w, custom_h)
    if err:
        return None, err
    return f"{custom_w}x{custom_h}", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GPT-Image-2", page_icon="🎨", layout="centered")
st.title("🎨 GPT-Image-2 — Generate & Edit")

tab_gen, tab_edit = st.tabs(["Generate", "Edit"])

# ---- Generate tab --------------------------------------------------------

with tab_gen:
    prompt = st.text_area("Prompt", height=120, key="gen_prompt")

    col1, col2 = st.columns(2)
    with col1:
        size_choice = st.selectbox("Size", POPULAR_SIZES + ["custom"], key="gen_size")
    with col2:
        quality = st.selectbox("Quality", QUALITY_OPTIONS, key="gen_quality")

    custom_w, custom_h = 1024, 1024
    if size_choice == "custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            custom_w = st.number_input("Width (px)", min_value=16, step=16, value=1024, key="gen_cw")
        with cc2:
            custom_h = st.number_input("Height (px)", min_value=16, step=16, value=1024, key="gen_ch")

    if st.button("Generate", type="primary", key="gen_btn"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            size_str, err = _resolve_size(size_choice, custom_w, custom_h)
            if err:
                st.error(err)
            else:
                with st.spinner("Generating image…"):
                    try:
                        result = client.images.generate(
                            model=DEPLOYMENT,
                            prompt=prompt,
                            size=size_str,
                            quality=quality,
                            n=1,
                        )
                        img_b64 = result.data[0].b64_json
                        img_bytes = base64.b64decode(img_b64)

                        st.image(img_bytes, use_container_width=True)

                        fname = f"generated_{datetime.now():%Y%m%d_%H%M%S}.png"
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=fname,
                            mime="image/png",
                        )
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

# ---- Edit tab -------------------------------------------------------------

with tab_edit:
    edit_prompt = st.text_area("Edit prompt", height=120, key="edit_prompt")
    uploaded = st.file_uploader(
        "Upload source image (PNG)",
        type=["png"],
        key="edit_upload",
    )

    col1e, col2e = st.columns(2)
    with col1e:
        size_choice_e = st.selectbox("Size", POPULAR_SIZES + ["custom"], key="edit_size")
    with col2e:
        quality_e = st.selectbox("Quality", QUALITY_OPTIONS, key="edit_quality")

    custom_we, custom_he = 1024, 1024
    if size_choice_e == "custom":
        cc1e, cc2e = st.columns(2)
        with cc1e:
            custom_we = st.number_input("Width (px)", min_value=16, step=16, value=1024, key="edit_cw")
        with cc2e:
            custom_he = st.number_input("Height (px)", min_value=16, step=16, value=1024, key="edit_ch")

    if st.button("Edit image", type="primary", key="edit_btn"):
        if not edit_prompt.strip():
            st.warning("Please enter an edit prompt.")
        elif uploaded is None:
            st.warning("Please upload a source image.")
        else:
            size_str_e, err_e = _resolve_size(size_choice_e, custom_we, custom_he)
            if err_e:
                st.error(err_e)
            else:
                with st.spinner("Editing image…"):
                    try:
                        img_file = io.BytesIO(uploaded.getvalue())
                        img_file.name = uploaded.name

                        result = client.images.edit(
                            model=DEPLOYMENT,
                            image=img_file,
                            prompt=edit_prompt,
                            size=size_str_e,
                            quality=quality_e,
                            n=1,
                        )
                        img_b64 = result.data[0].b64_json
                        img_bytes = base64.b64decode(img_b64)

                        st.image(img_bytes, use_container_width=True)

                        fname = f"edited_{datetime.now():%Y%m%d_%H%M%S}.png"
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=fname,
                            mime="image/png",
                        )
                    except Exception as exc:
                        st.error(f"Edit failed: {exc}")
