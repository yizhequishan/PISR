from __future__ import annotations

from pathlib import Path
from shutil import copy2

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
DOCX_PATH = OUT_DIR / "单次仿真截面邻域灵敏度解析研究报告.docx"
MD_PATH = OUT_DIR / "单次仿真截面邻域灵敏度解析研究报告.md"

FIG_BASE = ROOT / "outputs_mgn" / "fd_check" / "01_u_star_base.png"
FIG_DECOMP = ROOT / "outputs_mgn" / "fd_check" / "04_R_decomposition_fields.png"
FIG_KAPPA = ROOT / "outputs_mgn" / "fd_check" / "08_gradient_error_kappa_amplification.png"
FIG_OBS = ROOT / "outputs" / "p1_observability_nullspace" / "p1_observability.png"
FIG_DERIV = ROOT / "outputs" / "p2_analytic_derivative_loss" / "p2_value_vs_gradient.png"

NAVY = "17324D"
BLUE = "2E648C"
TEAL = "2A7F86"
GOLD = "B38A3E"
INK = "202A33"
MUTED = "66727C"
LIGHT = "EEF3F6"
LIGHT_BLUE = "E7F0F5"
LIGHT_GOLD = "F7F1E4"
LIGHT_RED = "F9E9E7"
WHITE = "FFFFFF"
GRID = "B7C4CC"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def set_table_borders(table, color=GRID, size=6) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)


def set_table_geometry(table, widths_in: list[float], indent_dxa=120) -> None:
    total_dxa = int(round(sum(widths_in) * 1440))
    grid_dxa = [int(round(w * 1440)) for w in widths_in]
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total_dxa))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    tbl_grid = table._tbl.tblGrid
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for value in grid_dxa:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(value))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            width = grid_dxa[min(idx, len(grid_dxa) - 1)]
            cell.width = Inches(width / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_run_font(run, size=None, bold=None, italic=None, color=INK, name="等线") -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = rgb(color)


def set_style_font(style, east_asia="等线", latin="Arial", size=10.5, color=INK, bold=None) -> None:
    style.font.name = latin
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    style.font.size = Pt(size)
    style.font.color.rgb = rgb(color)
    if bold is not None:
        style.font.bold = bold


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, size=10.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.widow_control = True

    for name, size, color, before, after in (
        ("Title", 27, NAVY, 0, 10),
        ("Subtitle", 13, MUTED, 0, 10),
        ("Heading 1", 16, NAVY, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 11.5, TEAL, 8, 4),
    ):
        style = styles[name]
        set_style_font(style, size=size, color=color, bold=name != "Subtitle")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    if "Report Caption" not in styles:
        cap = styles.add_style("Report Caption", WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles["Report Caption"]
    set_style_font(cap, size=9, color=MUTED)
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(8)
    cap.paragraph_format.keep_with_next = False

    if "Equation" not in styles:
        eq = styles.add_style("Equation", WD_STYLE_TYPE.PARAGRAPH)
    else:
        eq = styles["Equation"]
    set_style_font(eq, east_asia="宋体", latin="Cambria Math", size=11, color=INK)
    eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq.paragraph_format.space_before = Pt(5)
    eq.paragraph_format.space_after = Pt(7)
    eq.paragraph_format.keep_together = True

    if "Small Note" not in styles:
        note = styles.add_style("Small Note", WD_STYLE_TYPE.PARAGRAPH)
    else:
        note = styles["Small Note"]
    set_style_font(note, size=9, color=MUTED)
    note.paragraph_format.space_after = Pt(4)
    note.paragraph_format.line_spacing = 1.15


def set_page(section, landscape=False) -> None:
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = Inches(11)
        section.page_height = Inches(8.5)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)


def configure_header_footer(section, first_page=False) -> None:
    section.different_first_page_header_footer = first_page
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("单基线截面邻域灵敏度解析 | 技术研究报告")
    set_run_font(r, size=8.5, color=MUTED)
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run("2026-08-12  ·  ")
    set_run_font(r, size=8.5, color=MUTED)
    add_page_number(p)


def add_p(doc, text="", style=None, align=None, bold_prefix=None) -> None:
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        r2 = p.add_run(text[len(bold_prefix):])
        set_run_font(r1)
        set_run_font(r2)
    else:
        r = p.add_run(text)
        set_run_font(r)


def add_bullet(doc, text, level=0) -> None:
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.left_indent = Inches(0.28 + 0.22 * level)
    p.paragraph_format.first_line_indent = Inches(-0.16)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    set_run_font(r)


def add_number(doc, text) -> None:
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.32)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_run_font(r)


def add_callout(doc, label, text, fill=LIGHT_BLUE, accent=BLUE) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [6.62])
    set_table_borders(table, color=fill, size=2)
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(f"{label}  ")
    set_run_font(r, size=10.5, bold=True, color=accent)
    r = p.add_run(text)
    set_run_font(r, size=10.5, color=INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_equation(doc, text) -> None:
    p = doc.add_paragraph(style="Equation")
    r = p.add_run(text)
    set_run_font(r, name="Cambria Math", size=11, color=INK)


def add_table(doc, headers, rows, widths, font_size=8.8, header_fill=NAVY, first_col_bold=False,
              keep_rows_together=False):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        set_cell_shading(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_run_font(r, size=font_size, bold=True, color=WHITE)
    for ridx, row in enumerate(rows):
        row_obj = table.add_row()
        if keep_rows_together:
            tr_pr = row_obj._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
        cells = row_obj.cells
        if ridx % 2:
            for cell in cells:
                set_cell_shading(cell, "F7F9FA")
        for idx, text in enumerate(row):
            p = cells[idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if idx > 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(text))
            set_run_font(r, size=font_size, bold=(first_col_bold and idx == 0), color=INK)
    return table


def add_figure(doc, path: Path, width_in: float, caption: str, label: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    doc_pr = run._r.xpath(".//wp:docPr")
    if doc_pr:
        doc_pr[0].set("name", label)
        doc_pr[0].set("title", label)
        doc_pr[0].set("descr", caption)
    cap = doc.add_paragraph(style="Report Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"{label}  {caption}")
    set_run_font(r, size=9, color=MUTED)


def add_toc(doc) -> None:
    doc.add_heading("目录", level=1)
    entries = [
        "1  执行摘要",
        "2  研究问题、定义与口径",
        "3  截面邻域为什么有用，又为什么不够",
        "4  拟议方法：K 可识别，M 条件重建，T 组合输出",
        "5  与现有方向的相似性和差异",
        "6  当前项目证据：可行性、边界与风险",
        "7  实施方案与验证门槛",
        "8  结论与推荐表述",
        "附录 A  公式与误差分解",
        "附录 B  参考文献",
    ]
    for entry in entries:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.15)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(entry)
        set_run_font(r, size=10.5, color=INK)


def build_markdown() -> str:
    return r"""# 单次仿真截面邻域灵敏度解析：方法定位、可行性边界与实施路线

## 执行摘要

本报告建议把研究方向正式定位为：**单基线仿真、截面邻域数据驱动、物理约束的局部一阶响应重建**。在线推断阶段只使用一个参数点的一次基准仿真，提取目标截面及其上下游邻近截面；少量扰动算例仅用于离线验证，不计入在线推断成本。

最重要的结论是：邻近截面能补足流向空间导数、通量和局部算子信息，但不能自动提供参数方向的信息。对于几何参数，严格采用

$$T_a=M_a+K_a,$$

其中 $K_a$ 是由已知坐标映射和基准场空间梯度可解析获得的几何输运项；$M_a$ 是固定参考坐标下的内在流动重组，必须依赖切线/CSE 源项、局部边界闭合、参数扰动信息或强先验。故当前可交付的是 $K_a$ 及“闭合条件下的 $M_a$ 代理”，而非无条件的全局真梯度。

以下数字来自历史产物，尚未按 2026-09-08 修正协议重算：精确离散路径的 $T=M+K$ 闭合误差为 $1.70\times10^{-16}$；联合 $u+v$ 观测将合成字典秩从 $3/7$ 提高到 $6/7$，仅余流函数常数规范；旧 P2 在选式所用网格上的梯度 MSE 降低 68.4%，不能作为独立测试提升。同时，多算例数据也直接表明不能只用 $K$ 代替 $T$：半径参数的 $u$ 响应中，$|T|=13.84$、$|M|=27.01$、$|K|=38.06$，$K$ 与 $M$ 强烈抵消，只用 $K$ 的相对误差为 231%。

## 研究口径

建议区分三类“单次”：D0 为一次基线轨迹的被动后处理；D1 为一次 nonlinear primal 加一次或多次辅助线性 tangent/adjoint；D2 为参数扰动多算例。本文目标是 D0，现有伴随、CSE 和 AD 文献多数属于 D1。

令 $q=(\mathbf u,p)$、$R(q;a)=0$，真状态灵敏度 $s=q_a$ 满足 $R_qs=-R_a$，真 QoI 梯度为 $dJ/da=J_a+J_qs$。如果截面或测度随几何移动，还要加入截面 Jacobian、法向和边界速度等几何项。

## 方法框架

1. 在目标截面 $\Gamma_0$ 两侧选取 $\Gamma_\pm$，至少用三截面构成局部薄片；五截面用于二阶导数和步长审计。
2. 从单次仿真提取逐时刻 $u,v,p$，同时保留 $t\pm\Delta t$；时均场只用于均场代理，不替代瞬态切线方程。
3. 用流函数/散度约束拟合状态及空间导数，显式优化 H1/导数损失。
4. 由几何映射计算 $K_a=-\nabla q\cdot V_a$。
5. 在局部薄片求解闭合条件下的线性灵敏度代理 $\hat M_a$，并对人工边界迹、正则权重和薄片宽度做敏感性分析。
6. 输出 $\hat T_a=K_a+\hat M_a$，再通过 QoI 链式法则形成梯度代理；若闭合不唯一，报告区间而非单值。

## 与现有方向的关系

- 与 tangent/CSE 最相似：都使用线性化 N-S 算子和参数源项；差异在于本方法只在局部薄片上做条件重建，需显式报告边界闭合误差。
- 与伴随互补：伴随适合一个 QoI 对多参数，但仍是全域反向求解；截面型 QoI 并不使伴随局部化。
- 与 DMD/resolvent/SINDy 相似于单轨迹和可解释建模，但这些方法在固定参数下通常识别动力学、外力响应或 PDE 项，而不是参数导数。
- 与稀疏流场重建相似于可观测性和布点设计，但重建 $q(a_0)$ 不等于识别 $q_a(a_0)$。

## 验证门槛

- Gate 0：离散路径恒等式，relative L2 小于 $10^{-10}$；当前已通过。
- Gate 1：三/五截面或 $\delta,2\delta$ 的导数收敛，关键指标变化不超过 5–10%。
- Gate 2：代理的 H1/导数相对误差不超过 10%（探索期可放宽至 20%），相关系数不低于 0.95，符号一致率不低于 90%。
- Gate 3：薄片宽度、人工边界闭合、正则化和留一截面下 QoI 代理变化不超过 10%，排序及符号稳定。
- Gate 4：以少量 $\pm h,\pm2h$ 参数扰动或全域 tangent/adjoint 做盲测；有限差分必须出现步长平台。

## 推荐结论

本文不声称从单条基线轨迹中无条件恢复全局参数梯度；本文利用截面薄片数据识别可观测的运动学输运项与局部线性算子，并在显式人工边界闭合和物理正则下构造局部参数响应代理，通过闭合敏感性和少量离线扰动算例量化偏差与适用域。
"""


def build_docx() -> Document:
    doc = Document()
    configure_styles(doc)
    set_page(doc.sections[0])
    configure_header_footer(doc.sections[0], first_page=True)

    # Cover: editorial report pattern.
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("技术研究报告 · 方法论与实施路线")
    set_run_font(r, size=10, bold=True, color=GOLD)
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("单次仿真截面邻域灵敏度解析")
    set_run_font(r, size=27, bold=True, color=NAVY)
    p = doc.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("方法定位、可行性边界、现有方向相似性与验证路线")
    set_run_font(r, size=13, color=MUTED)

    add_callout(
        doc,
        "核心定位",
        "单基线仿真 + 目标截面及邻近截面 + 物理约束，先解析可观测的几何输运项 K，再在显式闭合条件下重建内在响应 M，最终形成 T = K + M 的局部响应代理。",
        fill=LIGHT_BLUE,
        accent=BLUE,
    )

    meta = doc.add_table(rows=4, cols=2)
    set_table_geometry(meta, [1.42, 5.20])
    set_table_borders(meta, color="D8E1E6", size=4)
    for i, (k, v) in enumerate(
        [
            ("研究对象", "二维非定常不可压圆柱绕流；截面型 QoI"),
            ("在线数据预算", "一个参数点的一次基准仿真；目标截面 + 邻近截面薄片"),
            ("验证原则", "单次推断、多算例离线验证；不把验证成本计入在线推断"),
            ("版本日期", "2026 年 8 月 12 日"),
        ]
    ):
        set_cell_shading(meta.cell(i, 0), LIGHT)
        for j, text in enumerate((k, v)):
            p = meta.cell(i, j).paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            set_run_font(r, size=9.5, bold=(j == 0), color=NAVY if j == 0 else INK)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(26)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Cylinder Flow PISR / OASIS-SR 研究线")
    set_run_font(r, size=9.5, color=MUTED)

    doc.add_page_break()
    add_toc(doc)

    doc.add_page_break()
    doc.add_heading("1  执行摘要", level=1)
    add_callout(
        doc,
        "结论先行",
        "该思路与连续灵敏度方程、局部物理约束重建和可观测性分析有清晰的学术亲缘，但它不是传统伴随/CSE的“局部免求解版”。邻近截面提供的是空间方向信息，不是参数方向信息。",
        fill=LIGHT_GOLD,
        accent=GOLD,
    )
    add_p(doc, "建议将研究方向正式命名为“单基线仿真、截面邻域数据驱动、物理约束的局部一阶响应重建”。在线推断阶段仅使用一个参数点的一次基准仿真；少量参数扰动算例只用于离线验证，不计入在线成本。")
    add_p(doc, "对于几何参数，报告必须锁定 T = K + M 的口径：K 是由坐标映射和基准场空间梯度可解析获得的运动学输运项；M 是固定参考坐标下的内在流动重组。目标截面及邻近截面能较好地支持 K、空间梯度、通量及局部算子，但不能无条件唯一识别 M。")
    add_p(doc, "当前项目已经证明这条路线具有方法价值，同时也暴露了必须正视的边界：")
    add_bullet(doc, "离散路径的 T=M+K 恒等式最大相对 L2 误差为 1.70×10⁻¹⁶，说明坐标、符号和归一化记账正确；这不等于物理灵敏度已经准确。")
    add_bullet(doc, "合成可观测性实验中，u 单分量观测秩为 3/7；联合 u+v 后为 6/7，仅余流函数常数规范自由度，说明邻近截面与互补分量确实能解除结构退化。")
    add_bullet(doc, "历史 P2 的选式网格梯度 MSE 降低 68.4%，函数值 MSE 增加 14.3%；选式与评价复用了同一网格，不能据此报告独立测试提升。新流程先在独立验证集选式，再冻结表达式评价测试集。")
    add_bullet(doc, "半径响应中 |T|=13.84、|M|=27.01、|K|=38.06，K 与 M 强烈抵消；只用 K 相对 T 的误差为 231%，直接否定“空间梯度即完整参数灵敏度”的表述。")
    add_callout(doc, "推荐主张", "当前可交付：截面 QoI、空间导数/通量、几何输运项 K、闭合条件下的 M/T 局部代理及不确定区间。当前不可交付：仅凭单条基线局部数据得到的无条件精确全局 dJ/da。", fill=LIGHT_RED, accent="A6463B")

    doc.add_heading("2  研究问题、定义与口径", level=1)
    doc.add_heading("2.1  先区分三种“单次仿真”", level=2)
    rows = [
        ("D0：单基线后处理", "一个参数点、一次基准轨迹；只读取截面/薄片数据，不解辅助线性系统", "本文目标"),
        ("D1：primal + 辅助线性求解", "一次非线性 primal，再按参数或 QoI 解 tangent/adjoint/AD", "多数文献所称 single-run"),
        ("D2：参数扰动多算例", "±h、±2h 或 ensemble；可作金标准与收敛验证", "本文仅作离线验证"),
    ]
    add_table(doc, ["级别", "计算与数据含义", "本报告位置"], rows, [1.55, 3.65, 1.42], font_size=9.1, first_col_bold=True)
    add_p(doc, "这一区分非常重要：伴随、连续灵敏度方程和完整求解器自动微分虽然避免了参数扰动重跑，但仍属于 D1；不能把它们的“单次”与 D0 的“只读一次局部数据”混为一谈。")

    doc.add_heading("2.2  真梯度的统一表达", level=2)
    add_p(doc, "令 q=(u,p)，控制方程写为 R(q;a)=0。真状态灵敏度 s=qₐ 满足线性化方程")
    add_equation(doc, "R_q(q₀;a₀) s = −R_a(q₀;a₀),")
    add_p(doc, "并带有初始条件与边界条件对参数 a 的导数。截面型 QoI 的真梯度为")
    add_equation(doc, "dJ/da = J_a + J_q s.")
    add_p(doc, "若截面 Γ、法向或积分测度随几何参数移动，还必须加入截面 Jacobian、法向变化及边界速度产生的 Hadamard/Leibniz 几何项。现有项目中的窗口平均绝对场值不是有符号 dJ/da；下一阶段应先固定 Γ 是物理固定截面，还是随 D 归一化移动的截面。")

    doc.add_heading("2.3  锁定 T、K、M 三层口径", level=2)
    add_p(doc, "设物理坐标与参考坐标的映射为 x=χ(ξ,a)，形状速度 Vₐ=∂ₐχ，参考场 q̃(ξ,a)=q(χ(ξ,a),a)。定义")
    add_equation(doc, "T_a = (∂q/∂a)|_x,    M_a = (∂q̃/∂a)|_ξ,    K_a = −∇_x q · V_a,")
    add_equation(doc, "T_a = M_a + K_a.")
    add_p(doc, "K 是运动学/坐标输运项，可由基准场空间梯度与已知几何映射解析计算；M 描述几何或物理参数改变后流动在随形坐标中的重新组织。只有在严格冻结图样或群对称假设成立时 M=0，K 才等于完整 T。若速度或时间还按随 a 变化的 U(a)、D(a) 归一化，应把幅值与时间重参数项另列，不能混入 K。")
    add_equation(doc, "K_R = −(2/D)(ξ q_ξ + η q_η),    K_cx = −(1/D)q_ξ,    K_cy = −(1/D)q_η.")

    doc.add_heading("3  截面邻域为什么有用，又为什么不够", level=1)
    doc.add_heading("3.1  邻近截面提供的可观测信息", level=2)
    add_p(doc, "以目标截面 Γ₀ 为中心，在同一时刻取上下游 Γ₋、Γ₊，三截面可用中心差分估计一阶和二阶流向导数；若再保留 t±Δt，可估计时间导数。结合截面内 η 方向采样，可构造散度、局部动量残差、黏性项与通量。")
    add_figure(doc, FIG_BASE, 4.05, "当前基准时均尾流条带，ξ∈[3,5]、33 个流向截面、64 个横向采样点。目标 ξ=4 的最近邻截面为 3.9375 与 4.0625。", "图 1")
    add_p(doc, "当前密集数据的实算比值为 ‖v‖/‖u‖=7.37%、‖∂ξu‖/‖∂ηu‖=17.30%、‖∂ξv‖/‖∂ηu‖=6.75%。流向变化不是可忽略小量，邻近截面能够提供真实的增量信息。")

    doc.add_heading("3.2  信息论边界：空间方向不等于参数方向", level=2)
    add_callout(doc, "不可辨识性命题", "固定 a=a₀ 的任何数量截面数据只约束解族在 a₀ 的取值。可以构造两个参数化解族在 a₀ 的全部观测完全相同，却具有任意不同的 ∂q/∂a|ₐ₀。因此，若不注入 R_a、扰动样本或强参数化先验，参数导数无法唯一识别。", fill=LIGHT_RED, accent="A6463B")
    add_p(doc, "把线性化算子按局部薄片内部 I 与人工边界 B 分块，局部状态灵敏度满足")
    add_equation(doc, "s_I = −L_II⁻¹ (R_{a,I} + L_IB s_B).")
    add_p(doc, "邻近 primal 截面能帮助估计 L、空间导数与局部源项，却不能给出被截掉外域传入的灵敏度边界迹 s_B；基准边界值 q_B 也不等于 s_B。不可压压力还通过椭圆约束和边界条件全局耦合。因此，局部薄片只能形成“给定闭合条件的条件解”，不能天然成为全域 CSE 的精确局部替代。")

    doc.add_heading("3.3  时均数据的额外边界", level=2)
    add_p(doc, "当前标准管线保留的是后半段 401 个时刻的时均 u、v；原始 COMSOL 文件虽包含 801 个时刻和压力，但压力与瞬态轨迹没有进入现有局部数据产物。若只把时均场代入瞬时灵敏度方程，会遗漏 Reynolds 应力及其参数导数，只能称作均场代理。稳定极限环应保留相位信息，并用相位对齐或 windowing 区分形状响应与相位漂移；混沌长期平均才进一步需要 LSS/NILSS 等方法。")

    doc.add_heading("4  拟议方法：K 可识别，M 条件重建，T 组合输出", level=1)
    doc.add_heading("4.1  数据几何与最小采样", level=2)
    add_number(doc, "以 Γ₀ 为目标截面，至少提取 Γ₋、Γ₀、Γ₊ 三截面；推荐同时保留 Γ₋₋、Γ₊₊，用于二阶导数和步长收敛审计。")
    add_number(doc, "每个截面保留 u、v、p 及坐标；非定常问题保留多个完整涡脱周期的逐时刻数据与 t±Δt，而不是只保留时均。")
    add_number(doc, "明确参数进入 PDE、边界与几何映射的方式，形成 R_a、形状速度 V_a、初/边条件导数。Re、ν、ρ 和边界条件必须写入元数据。")
    add_number(doc, "先固定 QoI 与截面运动学：物理固定 Γ 或随形 Γ；对移动截面显式处理积分测度和法向变化。")

    doc.add_heading("4.2  状态与导数重建层", level=2)
    add_p(doc, "以流函数 ψ 作为二维不可压结构变量，u=ψ_η、v=−ψ_ξ；联合 u+v 观测，显式加入 H1/解析导数损失。符号回归适合输出可解释表达式，MLP/PINN可作为高精度参照，但模型选择必须由值误差与导数误差共同决定。")
    add_equation(doc, "L_state = λ_u‖û−u‖² + λ_v‖v̂−v‖² + λ_H1‖∇q̂−∇q‖² + λ_R‖R(q̂)‖².")
    add_p(doc, "当前流函数 PISR 已经访问相邻 ξ 列并联合 u、v；这为三截面/五截面数据结构提供了代码基础。但现有符号 ψ 仍来自旧 16×64 网格，必须在当前 33×64 dense 网格上重训后才可作为最终模型。")

    doc.add_heading("4.3  K 层：单基线可直接解析", level=2)
    add_p(doc, "在 q̂ 及其空间导数通过 Gate 2 后，根据几何映射直接计算 K_a，并把离散路径 T=M+K 作为记账恒等式检查。K 是本方法中最坚实、最接近“单次仿真解析”的部分。")

    doc.add_heading("4.4  M 层：局部薄片切线代理", level=2)
    add_p(doc, "在局部薄片 Ωδ 上，以基准轨迹 q₀ 为系数，构造固定域示意方程")
    add_equation(doc, "∂t sδ +(u₀·∇)sδ +(sδ·∇)u₀ +∇σδ −νΔsδ = b_a(q₀),    ∇·sδ=0.")
    add_p(doc, "还需给定初值以及 Γ± 和侧边界上的灵敏度迹或通量。建议不把人工边界闭合隐藏成固定超参数，而是采用以下三种并行策略：")
    add_bullet(doc, "策略 A：从全域低成本线性/降阶求解得到界面迹，局部模型负责高分辨率与可解释重建。")
    add_bullet(doc, "策略 B：把界面迹设为低维未知，用物理残差、平滑性、上游因果和截面观测联合反演，并报告后验不确定性。")
    add_bullet(doc, "策略 C：定义可接受闭合集合 G，对每个 g∈G 求局部代理，输出 QoI 梯度区间 [min_g Ĝ, max_g Ĝ]；区间过宽则判定该截面不可辨识。")
    add_callout(doc, "方法名称", "建议使用“闭合条件下的局部切线代理（closure-conditioned local tangent proxy）”或“可观测性受限的物理约束局部灵敏度代理”，避免直接称为局部真 CSE。", fill=LIGHT_BLUE, accent=TEAL)

    doc.add_heading("4.5  T 与 QoI 输出层", level=2)
    add_equation(doc, "T̂_a = K̂_a + M̂_a,    Ĝ_a = J_a + J_q T̂_a + G_geometry + G_window.")
    add_p(doc, "报告应同时输出 K、M、T 以及各自误差/不确定性；不只给一个热图或平均绝对值。若没有独立扰动、全域 tangent 或 adjoint 真值，Ĝ 只能称“局部梯度代理/排序指标”。")

    # Landscape methods matrix.
    section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    set_page(section, landscape=True)
    configure_header_footer(section)
    doc.add_heading("5  与现有方向的相似性和差异", level=1)
    rows = [
        ("Tangent / CSE", "每参数一次全域正向线性 PDE", "真状态导数及任意 QoI", "最接近 M 的方程结构", "局部薄片不闭合；需 R_a、初边条件及全域传播"),
        ("连续/离散伴随", "每 QoI 一次全域反向线性 PDE", "一个 QoI 对多参数梯度", "截面 QoI 可作为伴随源", "截面目标不会使伴随局部化；需轨迹存储/重算"),
        ("前向/反向 AD", "对完整离散求解器求导", "离散计算图精确导数", "自动生成 tangent/adjoint", "参数须进入计算图；不是对输出快照自动求参数导数"),
        ("Windowing", "附加于 tangent/adjoint", "极限环平均灵敏度更稳", "适合低 Re 圆柱周期涡脱", "只减端点/相位误差，不创造参数信息"),
        ("LSS / NILSS", "长轨迹 + 多个切线/伴随", "混沌长期平均量导数", "都处理长期灵敏度病态", "non-intrusive 不等于局部；预算远超 D0"),
        ("DMDc / Koopman / resolvent", "固定参数时序；DMDc 需变化输入", "固定 a 动力学或 forcing→response", "可用局部传感、薄片时序", "forcing 响应≠Re/几何导数；通常无 ∂A/∂a"),
        ("SINDy / PDE-FIND", "时空数据 + 导数字典", "PDE 项与系数", "稀疏、可解释、可用物理库", "方程识别≠解的参数响应；少截面可能缺压力/法向项"),
        ("稀疏流场重建", "稀疏观测 + 训练库或 N-S 约束", "在先验流形上的状态", "可观测性、布点、局部补全", "重建 q(a₀)≠识别 q_a(a₀)"),
        ("局部控制体", "闭合控制面、通量/压力", "基准载荷或动量平衡", "截面数据可构造 QoI 与残差", "单个下游截面通常不闭合非定常力，更无 dJ/da"),
        ("本报告路线", "单基线局部薄片 + 显式闭合", "K 真值层 + M/T 条件代理", "兼顾局部数据、物理与解释性", "必须报告闭合敏感性；不承诺无条件全局真梯度"),
    ]
    add_table(doc, ["方向", "额外代价/数据", "严格输出", "相似处", "关键差异"], rows, [1.35, 2.05, 1.75, 2.05, 2.32], font_size=7.6, first_col_bold=True)
    add_p(doc, "术语提醒：one-shot optimization 指状态、伴随和设计变量同步迭代至 KKT 收敛，不等于“读取一条已存轨迹即可求灵敏度”；不建议借用 one-shot 作为本方法理论依据。", style="Small Note")

    # Back to portrait.
    section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    set_page(section, landscape=False)
    configure_header_footer(section)
    doc.add_heading("6  当前项目证据：可行性、边界与风险", level=1)
    doc.add_heading("6.1  P0：T=M+K 记账闭合，但 K 不能代替 T", level=2)
    add_figure(doc, FIG_DECOMP, 6.55, "半径参数下固定物理点的总响应 T、几何输运项 K、内在响应 M 及 K+M。K 与 M 的幅值均大于 T，存在显著抵消。", "图 2")
    rows = [
        ("R / u", "13.84", "27.01", "38.06", "2.717", "231.1%", "12.82%"),
        ("cₓ / u", "2.53", "1.37", "1.78", "0.719", "56.5%", "12.94%"),
        ("cᵧ / u", "4.67", "11.68", "9.50", "2.285", "262.3%", "13.20%"),
    ]
    add_table(doc, ["参数/分量", "mean|T|", "mean|M|", "mean|K|", "κ", "只用 K 对 T 误差", "数据梯度闭合误差"], rows, [1.08, .78, .78, .78, .62, 1.32, 1.26], font_size=8.3, first_col_bold=True)
    add_p(doc, "精确离散路径的闭合误差达到机器精度，只证明坐标映射、符号和插值路径一致。基准 CFD 空间梯度对经验 K 的误差虽可低至 4.72%（R/u），却被 κ=2.717 放大成 12.82% 的总闭合误差。灵敏度代理必须显式监控抵消条件数。")
    add_figure(doc, FIG_KAPPA, 5.65, "R/u 的输运误差经抵消条件数 κ 放大。符号 ψ-B 的输运误差约 16.9%，放大后总闭合误差约 45.8%；基准 CFD 数据梯度约 4.7%，放大后约 12.8%。", "图 3")

    doc.add_heading("6.2  P1：邻近截面和互补分量提高可辨识性", level=2)
    add_figure(doc, FIG_OBS, 6.55, "合成字典的观测奇异谱、最小范数系数恢复与稀疏互补 v 观测。该实验是结构机制证明，不是 CFD 参数灵敏度真值。", "图 4")
    add_p(doc, "u-only 观测矩阵秩为 3/7、零空间维数为 4；联合 u+v 后秩为 6/7，仅余常数规范。仅 3 个 D-optimal v 点（0.18% 的 v 网格）即可达到秩 6、系数误差 0.00114。建议把观测矩阵奇异谱、条件数和 D-optimal 布点作为截面选择的正式模块。")

    doc.add_heading("6.3  P2：值拟合与导数拟合是两个目标", level=2)
    add_figure(doc, FIG_DERIV, 6.55, "函数值误差和解析导数误差并非同一目标。导数感知选择可牺牲少量值精度，显著改善梯度。", "图 5")
    add_p(doc, "在历史 P2 的同一选式/评价网格上，解析导数/H1 选择将梯度 MSE 从 2.878×10⁻³ 降至 9.098×10⁻⁴（降低 68.4%），梯度相对 L2 从 9.63% 降至 5.41%；函数值 MSE 增加 14.3%。这些数值是选择后诊断，独立测试改进仍待新协议重跑。报告必须把 field/value、gradient/H1、PDE residual 三类误差分列。")

    doc.add_heading("6.4  当前版本与数据风险", level=2)
    add_bullet(doc, "标准 dense 条带是 33×64，但当前符号 ψ 与 φ_R 仍源自旧 16×64 网格；不能混作最终结果，需重训并重跑 Step 5–6。")
    add_bullet(doc, "原始基准数据含压力与 801 个时刻，当前 extractor 只保留时均 u、v；局部非定常 CSE 前必须重建瞬态 u、v、p 数据管线。")
    add_bullet(doc, "Re 在元数据中未提供；R± 算例的 Umax·D 近似恒定，可能意图固定 Re，但缺 ν 不能确认。报告不得沿用未经证实的“R 扰动改变 Re”。")
    add_bullet(doc, "P3 的 ±1%R 与 ±2%R 给出相反趋势，步长误差 75%–98%，时间块结果可变号且无网格独立性；该证据只能标“无结论”。")
    add_bullet(doc, "Step 5 已恢复 P0 所需字段并加入版本指纹；历史 NPZ 仍须重跑生成 v2 归档。本文数字和图表尚未用新协议重算，MLP 训练脚本也未进入根目录正式主线。")

    doc.add_heading("7  实施方案与验证门槛", level=1)
    doc.add_heading("7.1  六阶段实施路线", level=2)
    stages = [
        ("阶段 0：定义冻结", "固定 QoI、Γ 运动学、参数进入方式、单位与归一化；补齐 Re/ν/ρ/边界元数据。", "定义文档 + 数据字典"),
        ("阶段 1：瞬态薄片提取", "从原始宽表输出 u,v,p,t；支持 3/5/7 截面、多个 δ 与相位分箱。", "versioned NPZ/Parquet + 元数据"),
        ("阶段 2：状态/导数模型", "dense ψ-MLP 作精度基线；ψ-PISR 作可解释模型；联合 u+v、H1 与 PDE 残差。", "q̂、∇q̂、候选式与误差审计"),
        ("阶段 3：K 基线", "实现映射速度与移动截面几何项；输出 K 场、截面 QoI 贡献与 κ。", "单基线 K 报告"),
        ("阶段 4：局部 M 代理", "实现局部 tangent residual；比较边界闭合 A/B/C；输出单值或区间。", "M̂、T̂、闭合敏感性"),
        ("阶段 5：盲测与结论", "用 ±h、±2h 或全域 tangent/adjoint 对照；做网格、时间、相位与薄片宽度收敛。", "适用域与主张等级"),
    ]
    add_table(doc, ["阶段", "主要工作", "交付物"], stages, [1.55, 3.78, 1.29],
              font_size=8.7, first_col_bold=True, keep_rows_together=True)

    doc.add_heading("7.2  分层验证门槛", level=2)
    gates = [
        ("Gate 0", "定义与离散路径", "T=M+K rel-L2 < 10⁻¹⁰；单位、符号、映射和插值一致", "已通过：1.70×10⁻¹⁶"),
        ("Gate 1", "邻近截面导数", "δ、2δ 或三/五点 Richardson 收敛；关键指标变化 ≤5–10%", "尚未系统验证"),
        ("Gate 2", "状态/导数代理", "H1/导数 rel ≤10%（探索期 20%）；corr≥0.95；sign≥90%", "现符号梯度 35–44%，未通过"),
        ("Gate 3", "局部闭合稳健性", "薄片宽度、界面闭合、正则化、留一截面下 QoI 变化≤10%；排序/符号稳定", "待实现"),
        ("Gate 4", "独立真值盲测", "±h、±2h 出现步长平台；QoI rel≤10%；场 rel≤15–20%；corr≥0.95", "P3 75–98%，无结论"),
    ]
    add_table(doc, ["门槛", "对象", "通过标准", "当前状态"], gates, [.82, 1.45, 3.12, 1.23], font_size=8.4, first_col_bold=True)
    add_callout(doc, "验证原则", "“单次”描述在线推断成本，不限制研究阶段使用多算例建立可信度。最合理的论文设计是：单次推断，多次离线验证；验证算例严格隔离，不进入训练与参数选择。", fill=LIGHT_GOLD, accent=GOLD)

    doc.add_heading("7.3  误差预算与停止规则", level=2)
    add_p(doc, "局部真灵敏度限制 s|Ωδ 与代理 ŝδ 的误差 e 满足由数据、导数、算子、残差、边界和初值共同驱动的线性误差方程。可写成")
    add_equation(doc, "‖e‖ ≤ C_stab (E_data + E_deriv + E_operator + E_residual + E_boundary + E_initial).")
    add_p(doc, "QoI 层还需加入 E_J、移动截面几何误差和周期 window/相位误差；T 层满足 e_T=e_M+e_K。建议定义停止规则：任何 Gate 未通过，不把结果升级为梯度结论；只保留为响应代理或“无结论”。")

    doc.add_heading("8  结论与推荐表述", level=1)
    add_callout(doc, "最稳妥的创新定位", "本文不声称从单条基线轨迹中无条件恢复全局参数梯度；本文利用截面薄片数据识别可观测的运动学输运项与局部线性算子，并在显式人工边界闭合和物理正则下构造局部参数响应代理，同时通过闭合敏感性与少量离线扰动算例量化其偏差和适用域。", fill=LIGHT_BLUE, accent=BLUE)
    add_p(doc, "这一定位与现有方向“相似但不重复”：方程结构上最接近 tangent/CSE，数据策略上接近稀疏可观测性与物理约束重建，表达层上接近 SINDy/PISR；真正的研究贡献应落在局部截面薄片的可观测性设计、人工边界闭合、不确定区间和单次在线推断成本，而不是宣称局部数据天然等价于全域真梯度。")
    add_p(doc, "按当前证据，建议近期论文/汇报采用三级主张：")
    add_bullet(doc, "A级（已支撑）：单基线截面邻域可恢复空间导数与几何输运项 K，并可量化代理梯度误差与 κ 放大。")
    add_bullet(doc, "B级（待实现）：在显式闭合条件下重建局部 M/T 代理，且对薄片宽度、闭合与相位稳定。")
    add_bullet(doc, "C级（通过 Gate 4 后）：对预注册参数/QoI，在给定工况域内近似预测 dJ/da；不得外推为一般形状梯度。")

    doc.add_heading("附录 A  公式与项目数据要点", level=1)
    add_p(doc, "项目当前 dense 局部条带：D=0.1 m，圆心 (0.2,0.2) m，ξ∈[3,5]，33×64，Δξ=0.0625，Δη≈0.0603；基准原始文件 6361 节点、801 时刻，标准管线取索引 400:801 做时均。")
    add_p(doc, "当前核心几何输运公式对无量纲标量/速度分量 q 成立；若 q 自身的无量纲化标度 U(a) 随参数变化，还需把标度导数单独加入。移动截面 QoI 还需显式几何积分项。")
    add_p(doc, "周期时间平均建议使用归一化窗：J_w=(1/T)∫₀ᵀw(t/T)j(t)dt，且 ∫₀¹w(τ)dτ=1。windowing 改善有限窗相位/端点误差，不替代 tangent/adjoint/CSE 中的参数源项。")

    doc.add_heading("附录 B  参考文献", level=1)
    refs = [
        "[1] Borggaard, J. T.; Burns, J. A. A PDE Sensitivity Equation Method for Optimal Aerodynamic Design. Journal of Computational Physics, 1997. DOI: 10.1006/jcph.1997.5743.",
        "[2] Giles, M. B.; Pierce, N. A. An Introduction to the Adjoint Approach to Design. Flow, Turbulence and Combustion, 2000. DOI: 10.1023/A:1011430410075.",
        "[3] Pelletier, D. et al. The Sensitivity Equation Method in Fluid Mechanics. European Journal of Computational Mechanics, 2008. DOI: 10.3166/remn.17.31-61.",
        "[4] Ilinca, F.; Pelletier, D.; Borggaard, J. A Continuous Second-Order Sensitivity Equation Method for Time-Dependent Incompressible Laminar Flows. IJNMF, 2007. DOI: 10.1002/fld.1477.",
        "[5] Krakos, J. A. et al. Sensitivity Analysis of Limit Cycle Oscillations. Journal of Computational Physics, 2012. DOI: 10.1016/j.jcp.2012.01.001.",
        "[6] Wang, Q.; Hu, R.; Blonigan, P. Least Squares Shadowing Sensitivity Analysis of Chaotic Limit Cycle Oscillations. JCP, 2014. DOI: 10.1016/j.jcp.2014.03.002.",
        "[7] Ni, A.; Wang, Q. Sensitivity Analysis on Chaotic Dynamical Systems by NILSS. JCP, 2017. DOI: 10.1016/j.jcp.2017.06.033.",
        "[8] Proctor, J. L.; Brunton, S. L.; Kutz, J. N. Dynamic Mode Decomposition with Control. SIAM JADS, 2016. DOI: 10.1137/15M1013857.",
        "[9] Herrmann, B. et al. Data-Driven Resolvent Analysis. Journal of Fluid Mechanics, 2021. DOI: 10.1017/jfm.2021.337.",
        "[10] Brunton, S. L.; Proctor, J. L.; Kutz, J. N. Discovering Governing Equations from Data by SINDy. PNAS, 2016. DOI: 10.1073/pnas.1517384113.",
        "[11] Rudy, S. H. et al. Data-Driven Discovery of Partial Differential Equations. Science Advances, 2017. DOI: 10.1126/sciadv.1602614.",
        "[12] Venturi, D.; Karniadakis, G. E. Gappy Data and Reconstruction Procedures for Flow Past a Cylinder. JFM, 2004. DOI: 10.1017/S0022112004001338.",
        "[13] Loiseau, J.-C.; Noack, B. R.; Brunton, S. L. Sparse Reduced-Order Modelling: Sensor-Based Dynamics to Full-State Estimation. JFM, 2018. DOI: 10.1017/jfm.2018.147.",
        "[14] Marquet, O.; Sipp, D.; Jacquin, L. Sensitivity Analysis and Passive Control of Cylinder Flow. JFM, 2008. DOI: 10.1017/S0022112008003662.",
        "[15] Griewank, A.; Walther, A. Algorithm 799: revolve. ACM TOMS, 2000. DOI: 10.1145/347837.347846.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.22)
        p.paragraph_format.first_line_indent = Inches(-0.22)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.12
        r = p.add_run(ref)
        set_run_font(r, size=8.7, color=INK)

    return doc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(build_markdown(), encoding="utf-8")
    doc = build_docx()
    doc.core_properties.title = "单次仿真截面邻域灵敏度解析：方法定位、可行性边界与实施路线"
    doc.core_properties.subject = "圆柱绕流单基线截面邻域灵敏度解析研究"
    doc.core_properties.author = "Cylinder Flow PISR / OASIS-SR"
    doc.core_properties.keywords = "灵敏度; 截面; CSE; PISR; 圆柱绕流; 可观测性"
    doc.save(DOCX_PATH)
    print(DOCX_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
