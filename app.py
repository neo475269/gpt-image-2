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
LLM_DEPLOYMENT = "gpt-5.4"

REFINE_SYSTEM_PROMPT = """You are an expert prompt engineer for the gpt-image-2 image generation model.

Your job is to take the user's image prompt and produce an improved, highly detailed version that will yield a better image. You must:

1. **Plan layout**: Decide composition, perspective, framing, and spatial arrangement of elements.
2. **Specify typography**: If the image contains text, specify font style (serif, sans-serif, handwritten, etc.), weight, size relative to the image, colour, and placement.
3. **Enrich details**: Add art style, lighting, colour palette, mood, and texture descriptions.
4. **Research when needed**: If the prompt references real people, places, events, brands, trending topics, or anything that benefits from up-to-date or deeper factual information, search the internet to get accurate details and incorporate them.
5. **Preserve intent**: Never change the user's core idea. Only enhance it.

Return ONLY the refined prompt text. No explanations, no markdown fences, no preamble."""


def refine_prompt(user_prompt: str, mode: str = "generate") -> str:
    """Call gpt-5.4 with thinking + web search to refine an image prompt."""
    context = (
        "I want to GENERATE a new image."
        if mode == "generate"
        else "I want to EDIT an existing image."
    )
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        input=[
            {"role": "developer", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nMy prompt:\n{user_prompt}"},
        ],
        tools=[{"type": "web_search_preview"}],
        reasoning={"effort": "medium"},
    )
    return response.output_text


# ---------------------------------------------------------------------------
# Size / quality helpers
# ---------------------------------------------------------------------------

SIZE_PRESETS = {
    "Auto": "auto",
    "1K Square (1024×1024)": "1024x1024",
    "1K × 1.5K Portrait (1024×1536)": "1024x1536",
    "1.5K × 1K Landscape (1536×1024)": "1536x1024",
    "2K Square (2048×2048)": "2048x2048",
    "2K Portrait 3:4 (1536×2048)": "1536x2048",
    "2K Portrait 9:16 (1152×2048)": "1152x2048",
    "2K Landscape 3:4 (2048×1536)": "2048x1536",
    "2K Landscape 9:16 (2048×1152)": "2048x1152",
    "4K Square (2880×2880)": "2880x2880",
    "4K Portrait 3:4 (2304×3072)": "2304x3072",
    "4K Portrait 9:16 (2160×3840)": "2160x3840",
    "4K Landscape 3:4 (3072×2304)": "3072x2304",
    "4K Landscape 9:16 (3840×2160)": "3840x2160",
}

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
    if choice == "Custom":
        err = _validate_custom_size(custom_w, custom_h)
        if err:
            return None, err
        return f"{custom_w}x{custom_h}", None
    return SIZE_PRESETS[choice], None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GPT-Image-2", page_icon="🎨", layout="centered")
st.title("🎨 GPT-Image-2 — Generate & Edit")

# Persist results across reruns
for _key in ("gen_result", "gen_fname", "edit_result", "edit_fname"):
    if _key not in st.session_state:
        st.session_state[_key] = None

tab_gen, tab_edit = st.tabs(["Generate", "Edit"])

# ---- Generate tab --------------------------------------------------------

with tab_gen:
    prompt = st.text_area("Prompt", height=120, key="gen_prompt")
    use_refine = st.toggle("✨ Refine prompt with GPT-5.4", value=True, key="gen_refine")

    col1, col2 = st.columns(2)
    with col1:
        size_choice = st.selectbox("Size", list(SIZE_PRESETS.keys()) + ["Custom"], key="gen_size")
    with col2:
        quality = st.selectbox("Quality", QUALITY_OPTIONS, key="gen_quality")

    custom_w, custom_h = 1024, 1024
    if size_choice == "Custom":
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
                final_prompt = prompt
                if use_refine:
                    with st.spinner("Refining prompt with GPT-5.4 (thinking + web search)…"):
                        try:
                            final_prompt = refine_prompt(prompt, mode="generate")
                            st.session_state.gen_refined_prompt = final_prompt
                        except Exception as exc:
                            st.warning(f"Prompt refinement failed, using original: {exc}")

                with st.spinner("Generating image…"):
                    try:
                        result = client.images.generate(
                            model=DEPLOYMENT,
                            prompt=final_prompt,
                            size=size_str,
                            quality=quality,
                            n=1,
                        )
                        img_b64 = result.data[0].b64_json
                        img_bytes = base64.b64decode(img_b64)
                        st.session_state.gen_result = img_bytes
                        st.session_state.gen_fname = f"generated_{datetime.now():%Y%m%d_%H%M%S}.png"
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

    if st.session_state.get("gen_refined_prompt"):
        with st.expander("🔍 Refined prompt", expanded=False):
            st.markdown(st.session_state.gen_refined_prompt)

    if st.session_state.gen_result is not None:
        st.image(st.session_state.gen_result, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download",
                data=st.session_state.gen_result,
                file_name=st.session_state.gen_fname,
                mime="image/png",
                key="gen_dl",
            )
        with c2:
            if st.button("🔄 Start over", key="gen_reset"):
                st.session_state.gen_result = None
                st.session_state.gen_fname = None
                st.session_state.gen_refined_prompt = None
                st.rerun()

# ---- Edit tab -------------------------------------------------------------

with tab_edit:
    edit_prompt = st.text_area("Edit prompt", height=120, key="edit_prompt")
    uploaded_files = st.file_uploader(
        "Upload source images (PNG, up to 10)",
        type=["png"],
        accept_multiple_files=True,
        key="edit_upload",
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.warning("Maximum 10 images allowed. Only the first 10 will be used.")
            uploaded_files = uploaded_files[:10]
        cols = st.columns(min(len(uploaded_files), 5))
        for i, f in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                st.image(f, caption=f.name, use_container_width=True)

    col1e, col2e = st.columns(2)
    with col1e:
        size_choice_e = st.selectbox("Size", list(SIZE_PRESETS.keys()) + ["Custom"], key="edit_size")
    with col2e:
        quality_e = st.selectbox("Quality", QUALITY_OPTIONS, key="edit_quality")

    custom_we, custom_he = 1024, 1024
    if size_choice_e == "Custom":
        cc1e, cc2e = st.columns(2)
        with cc1e:
            custom_we = st.number_input("Width (px)", min_value=16, step=16, value=1024, key="edit_cw")
        with cc2e:
            custom_he = st.number_input("Height (px)", min_value=16, step=16, value=1024, key="edit_ch")

    use_refine_e = st.toggle("✨ Refine prompt with GPT-5.4", value=True, key="edit_refine")

    if st.button("Edit image", type="primary", key="edit_btn"):
        if not edit_prompt.strip():
            st.warning("Please enter an edit prompt.")
        elif not uploaded_files:
            st.warning("Please upload at least one source image.")
        else:
            size_str_e, err_e = _resolve_size(size_choice_e, custom_we, custom_he)
            if err_e:
                st.error(err_e)
            else:
                final_edit_prompt = edit_prompt
                if use_refine_e:
                    with st.spinner("Refining prompt with GPT-5.4 (thinking + web search)…"):
                        try:
                            final_edit_prompt = refine_prompt(edit_prompt, mode="edit")
                            st.session_state.edit_refined_prompt = final_edit_prompt
                        except Exception as exc:
                            st.warning(f"Prompt refinement failed, using original: {exc}")

                with st.spinner("Editing image…"):
                    try:
                        image_inputs = []
                        for f in uploaded_files:
                            buf = io.BytesIO(f.getvalue())
                            buf.name = f.name
                            image_inputs.append(buf)

                        result = client.images.edit(
                            model=DEPLOYMENT,
                            image=image_inputs if len(image_inputs) > 1 else image_inputs[0],
                            prompt=final_edit_prompt,
                            size=size_str_e,
                            quality=quality_e,
                            n=1,
                        )
                        img_b64 = result.data[0].b64_json
                        img_bytes = base64.b64decode(img_b64)
                        st.session_state.edit_result = img_bytes
                        st.session_state.edit_fname = f"edited_{datetime.now():%Y%m%d_%H%M%S}.png"
                    except Exception as exc:
                        st.error(f"Edit failed: {exc}")

    if st.session_state.get("edit_refined_prompt"):
        with st.expander("🔍 Refined prompt", expanded=False):
            st.markdown(st.session_state.edit_refined_prompt)

    if st.session_state.edit_result is not None:
        st.image(st.session_state.edit_result, use_container_width=True)
        c1e, c2e = st.columns(2)
        with c1e:
            st.download_button(
                "⬇️ Download",
                data=st.session_state.edit_result,
                file_name=st.session_state.edit_fname,
                mime="image/png",
                key="edit_dl",
            )
        with c2e:
            if st.button("🔄 Start over", key="edit_reset"):
                st.session_state.edit_result = None
                st.session_state.edit_fname = None
                st.session_state.edit_refined_prompt = None
                st.rerun()
