import streamlit as st
import fitz  # pymupdf
import statistics
import random
import io
import tempfile

# Constants
MM_TO_PT = 2.83465

@st.cache_data(show_spinner="Analyzing PDF...")
def analyze_pdf_stats(file_bytes):
    """
    Analyze the PDF to find median font size and content geometry.
    Samples up to 100 random pages.
    """
    # Open document from bytes inside the cached function
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    total_pages = len(doc)
    # User manually requested 100 samples in recent edit, preserving that.
    num_samples = min(100, total_pages)
    
    sample_indices = sorted(random.sample(range(total_pages), num_samples))
    
    font_sizes = []
    content_rects = []
    page_rects = []
    
    # Progress bar doesn't work well inside cached function (it runs once).
    # We can remove it or keep it for the first run.
    
    for i in sample_indices:
        page = doc[i]
        page_rects.append(page.rect)
        
        text_dict = page.get_text("dict")
        
        # Bounding box of content (text + images)
        page_content_rect = fitz.Rect()
        valid_content_found = False
        
        for block in text_dict.get("blocks", []):
            block_rect = fitz.Rect(block["bbox"])
            if not valid_content_found:
                page_content_rect = block_rect
                valid_content_found = True
            else:
                page_content_rect |= block_rect # Union
            
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = span["size"]
                        if size > 0:
                            font_sizes.append(size)
                            
        if valid_content_found:
            content_rects.append(page_content_rect)
    
    # Calculate stats
    median_font = statistics.median(font_sizes) if font_sizes else 0.0
    
    # Average page dimensions
    # Return simple float values to ensure pickling for cache works
    avg_page_w = 420.0
    avg_page_h = 595.0
    
    if page_rects:
        avg_page_w = statistics.mean([r.width for r in page_rects])
        avg_page_h = statistics.mean([r.height for r in page_rects])

    # Average Content Rect dimensions
    avg_content_w = avg_page_w
    avg_content_h = avg_page_h
    
    if content_rects:
        avg_content_w = statistics.mean([r.width for r in content_rects])
        avg_content_h = statistics.mean([r.height for r in content_rects])
        
    doc.close()
        
    return {
        "median_font": median_font,
        "avg_page_w": avg_page_w,
        "avg_page_h": avg_page_h,
        "avg_content_w": avg_content_w,
        "avg_content_h": avg_content_h,
        "sample_count": num_samples,
        "has_text": bool(font_sizes)
    }

def calculate_auto_scale(doc_stats, target_font_size, min_margin_mm, gap_mm):
    """
    Calculates the optimal scale based on constraints.
    """
    median_font = doc_stats["median_font"]
    
    # Use simple geometry values
    content_w = doc_stats["avg_content_w"]
    content_h = doc_stats["avg_content_h"]
    
    min_margin_pt = min_margin_mm * MM_TO_PT
    gap_pt = gap_mm * MM_TO_PT
    
    slot_width = (842.0 - gap_pt) / 2
    slot_height = 595.0
    
    if slot_width <= 0:
        return 1.0, 1.0, "Gap too large", slot_width
    
    max_content_w = slot_width - (2 * min_margin_pt)
    max_content_h = slot_height - (2 * min_margin_pt)
    
    if max_content_w <= 0 or max_content_h <= 0:
        return 1.0, 1.0, "Margins/Gap too large for page", slot_width

    # Geometric constraints
    scale_w = max_content_w / content_w if content_w > 0 else 999
    scale_h = max_content_h / content_h if content_h > 0 else 999
    max_geometric_scale = min(scale_w, scale_h)
    
    # Font constraints
    if median_font > 0 and target_font_size > 0:
        target_scale = target_font_size / median_font
    else:
        target_scale = 1.0 
        
    # Clamping
    final_scale = min(target_scale, max_geometric_scale)
    
    return final_scale, median_font * final_scale, slot_width

def create_imposed_page(writer, doc, p_left_idx, p_right_idx, scale, slot_width, gap_pt, vertical_shift_pt, show_cut_marks, cut_mark_padding_pt, target_size=(842, 595)):
    """
    Creates a single A4 landscape page with p_left and p_right.
    """
    out_page = writer.new_page(width=target_size[0], height=target_size[1])
    
    # Slot Centers with Vertical Shift
    # Default center is target_size[1] / 2
    # Shift: +Down, -Up
    base_center_y = target_size[1] / 2
    shifted_center_y = base_center_y + vertical_shift_pt
    
    left_slot_center_x = slot_width / 2
    left_slot_center_y = shifted_center_y
    
    right_slot_center_x = slot_width + gap_pt + (slot_width / 2)
    right_slot_center_y = shifted_center_y
    
    # Left Page
    if p_left_idx < len(doc):
        src_page = doc[p_left_idx]
        p_w = src_page.rect.width * scale
        p_h = src_page.rect.height * scale
        
        rect_left = fitz.Rect(
            left_slot_center_x - p_w/2,
            left_slot_center_y - p_h/2,
            left_slot_center_x + p_w/2,
            left_slot_center_y + p_h/2
        )
        out_page.show_pdf_page(rect_left, doc, p_left_idx)

    # Right Page
    if p_right_idx < len(doc):
        src_page = doc[p_right_idx]
        p_w = src_page.rect.width * scale
        p_h = src_page.rect.height * scale
        
        rect_right = fitz.Rect(
            right_slot_center_x - p_w/2,
            right_slot_center_y - p_h/2,
            right_slot_center_x + p_w/2,
            right_slot_center_y + p_h/2
        )
        out_page.show_pdf_page(rect_right, doc, p_right_idx)
        
    if show_cut_marks:
        # Draw cut marks at the center of the sheet (or gap center)
        center_x = target_size[0] / 2
        
        shape = out_page.new_shape()
        
        mark_len = 14 # Length of the mark
        
        # Top Mark
        # Start at padding, go inwards
        y1_top = cut_mark_padding_pt
        y2_top = cut_mark_padding_pt + mark_len
        shape.draw_line((center_x, y1_top), (center_x, y2_top)) 
        
        # Bottom Mark
        y1_bot = target_size[1] - cut_mark_padding_pt
        y2_bot = y1_bot - mark_len
        shape.draw_line((center_x, y1_bot), (center_x, y2_bot))
        
        shape.finish(color=(0, 0, 0), width=0.3) # Thinner black line
        shape.commit()
        
    return out_page

def generate_preview_image(doc, scale, slot_width, gap_pt, vertical_shift_pt, sheet_index, show_cut_marks, cut_mark_padding_pt):
    """
    Generates a preview image of a specific imposed sheet.
    """
    out_pdf = fitz.open()
    
    total_pages = len(doc)
    n = total_pages
    if n % 2 != 0:
        n += 1
    
    half = n // 2
    
    if half == 0:
        return None
        
    # Clamp sheet index
    if sheet_index < 0: sheet_index = 0
    if sheet_index >= half: sheet_index = half - 1
        
    p_left = sheet_index
    p_right = half + sheet_index
    
    page = create_imposed_page(out_pdf, doc, p_left, p_right, scale, slot_width, gap_pt, vertical_shift_pt, show_cut_marks, cut_mark_padding_pt)
    
    # --- Preview Visual Overlays ---
    shape = page.new_shape()
    
    # 1. Paper Edge Border (Red)
    rect = page.rect
    shape.draw_rect(rect)
    shape.finish(color=(1, 0, 0), width=1)
    
    # 2. Non-Printable Area (Dark gray dashed box, 5mm from edge)
    non_print_pt = 5 * MM_TO_PT
    safe_rect = fitz.Rect(non_print_pt, non_print_pt, 842 - non_print_pt, 595 - non_print_pt)
    shape.draw_rect(safe_rect)
    # Use simpler dash pattern and explicit overlay
    shape.finish(color=(0.3, 0.3, 0.3), width=1.0, dashes=[4]) 
    
    # 3. Gap Indicator (Blue shaded area or lines)
    if gap_pt > 0:
        gap_x1 = slot_width
        gap_x2 = slot_width + gap_pt
        
        # Draw lines and fill
        shape.draw_rect(fitz.Rect(gap_x1, 0, gap_x2, 595))
        shape.finish(color=(0, 1, 1), width=0, fill=(0.9, 0.9, 1)) 
        
        # Re-draw lines for emphasis
        shape.draw_line((gap_x1, 0), (gap_x1, 595))
        shape.draw_line((gap_x2, 0), (gap_x2, 595))
        shape.finish(color=(0, 0, 1), width=0.5)
        
    shape.commit(overlay=True)

    pix = page.get_pixmap(dpi=72)
    data = pix.tobytes("png")
    out_pdf.close()
    return data

def generate_imposed_pdf(doc, scale, slot_width, gap_pt, vertical_shift_pt, show_cut_marks, cut_mark_padding_pt):
    out_pdf = fitz.open()
    total_pages = len(doc)
    n = total_pages
    if n % 2 != 0: n += 1
    half = n // 2
    
    for i in range(half):
        p_left = i
        p_right = half + i
        create_imposed_page(out_pdf, doc, p_left, p_right, scale, slot_width, gap_pt, vertical_shift_pt, show_cut_marks, cut_mark_padding_pt)
        
    return out_pdf.tobytes()


# --- Streamlit UI ---

st.set_page_config(page_title="PDF Imposer", layout="wide")

st.title("📄 Smart PDF Imposer (Cut & Stack)")
st.markdown("Upload an A5 PDF to impose it onto A4 paper.")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload PDF Book", type="pdf")

# Mode Selection
st.sidebar.subheader("General Settings")
mode = st.sidebar.radio("Optimization Mode", ["Auto (Smart Scale)", "Manual Scale"], index=0)

gap_mm = st.sidebar.number_input("Gap between pages (mm)", value=0.0, step=1.0)
vertical_shift_mm = st.sidebar.number_input("Vertical Shift (mm)", value=0.0, step=1.0, help="Positive moves content DOWN, Negative moves UP")
show_cut_marks = st.sidebar.checkbox("Add Cut Marks", value=True)

cut_mark_padding_mm = 20.0
if show_cut_marks:
    cut_mark_padding_mm = st.sidebar.slider("Cut Mark Padding (mm)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)

# Conditional Inputs
final_scale = 1.0
effective_font = 0.0

# Placeholders for analysis stats
stats = None
slot_width = (842.0 - (gap_mm * MM_TO_PT)) / 2

if uploaded_file:
    file_bytes = uploaded_file.read() 
    
    # Open doc for LIVE usage (preview/generation) - fast operation
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    # Optimized Analysis with Caching
    # Passes bytes to cached function
    stats = analyze_pdf_stats(file_bytes)
    
    if mode == "Auto (Smart Scale)":
        target_font_size = st.sidebar.number_input("Target Font Size (pt)", value=11.0, step=0.5)
        min_margin_mm = st.sidebar.number_input("Min Margin (Text to Edge) (mm)", value=10.0, step=1.0)
        
        final_scale, effective_font, slot_width = calculate_auto_scale(stats, target_font_size, min_margin_mm, gap_mm)
        
        if stats['has_text']:
            st.sidebar.success(f"Detected Median: **{stats['median_font']:.1f} pt** (Sampled {stats['sample_count']} pages)")
            
    else: # Manual Mode
        st.sidebar.info("Manual Mode: Geometric safety limits are ignored.")
        manual_scale = st.sidebar.number_input("Scale Factor", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
        final_scale = manual_scale
        effective_font = stats['median_font'] * final_scale
        
    # Preview Controls
    total_input_pages = len(doc)
    imposed_sheets = (total_input_pages + 1) // 2
    
    st.sidebar.header("Preview")
    preview_sheet_idx = st.sidebar.number_input(
        f"Preview Sheet (1-{imposed_sheets})", 
        min_value=1, 
        max_value=imposed_sheets, 
        value=1
    ) - 1
    
    # Main Display
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Layout Details")
        st.markdown(f"**Mode:** {mode}")
        st.markdown(f"**Scale Factor:** `{final_scale:.2f}x`")
        if stats['has_text']:
            st.markdown(f"**Effective Font Size:** `{effective_font:.1f} pt`")
            if mode == "Auto (Smart Scale)":
                 diff = effective_font - target_font_size
                 if abs(diff) > 0.1:
                    st.info(f"Target was {target_font_size} pt. Clamped to safe margins.")
        
        st.markdown(f"**Gap:** {gap_mm} mm")
        st.markdown(f"**Vertical Shift:** {vertical_shift_mm} mm")
        st.markdown(f"**Cut Marks:** {'Yes' if show_cut_marks else 'No'} (Pad: {cut_mark_padding_mm}mm)")
        
        if st.button("Download Imposed PDF", type="primary"):
            with st.spinner("Generating full PDF..."):
                pdf_bytes = generate_imposed_pdf(doc, final_scale, slot_width, gap_mm * MM_TO_PT, vertical_shift_mm * MM_TO_PT, show_cut_marks, cut_mark_padding_mm * MM_TO_PT)
            st.download_button(
                label="Save imposed_book.pdf",
                data=pdf_bytes,
                file_name="imposed_book.pdf",
                mime="application/pdf"
            )

    with col2:
        st.subheader(f"Live Preview (Sheet {preview_sheet_idx + 1})")
        preview_png = generate_preview_image(doc, final_scale, slot_width, gap_mm * MM_TO_PT, vertical_shift_mm * MM_TO_PT, preview_sheet_idx, show_cut_marks, cut_mark_padding_mm * MM_TO_PT)
        
        if preview_png:
            try:
                # Try new API first to avoid warnings
                st.image(preview_png, caption="A4 Preview (Red=Edge, Blue=Gap, Gray=Safe Area)", width="stretch")
            except Exception:
                st.image(preview_png, caption="A4 Preview (Red=Edge, Blue=Gap, Gray=Safe Area)")
            
else:
    st.info("Please upload a PDF file to begin.")
