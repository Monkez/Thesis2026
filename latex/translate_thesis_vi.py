import json
import re
import shutil
import time
from pathlib import Path

from deep_translator import GoogleTranslator


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "latex_vi"
FILES = [
    "thesis.tex",
    "body_from_docx.tex",
    "expanded_ch1.tex",
    "expanded_ch1_ml.tex",
    "expanded_ch2.tex",
    "expanded_ch3.tex",
    "expanded_demapper.tex",
    "expanded_validation.tex",
]
CACHE_PATH = OUT / "translation_cache_vi.json"


SKIP_PREFIXES = (
    "\\documentclass",
    "\\usepackage",
    "\\setmainfont",
    "\\usetikzlibrary",
    "\\geometry",
    "\\onehalfspacing",
    "\\setlength",
    "\\emergencystretch",
    "\\clubpenalty",
    "\\widowpenalty",
    "\\displaywidowpenalty",
    "\\brokenpenalty",
    "\\finalhyphendemerits",
    "\\hyphenpenalty",
    "\\exhyphenpenalty",
    "\\raggedbottom",
    "\\setcounter",
    "\\graphicspath",
    "\\hypersetup",
    "\\titleformat",
    "\\titlespacing",
    "\\renewcommand",
    "\\newcommand",
    "\\makeatletter",
    "\\makeatother",
    "\\pagestyle",
    "\\fancy",
    "\\renewenvironment",
    "\\captionsetup",
    "\\begin{document}",
    "\\end{document}",
    "\\input",
    "\\includegraphics",
    "\\label",
    "\\addcontentsline",
    "\\manualtocline",
    "\\thispagestyle",
    "\\pagenumbering",
    "\\newpage",
    "\\clearpage",
    "\\vspace",
    "\\hspace",
    "\\rule",
    "\\chapter*",
    "\\bibitem",
)

MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "split",
}

RAW_ENVS = {
    "tikzpicture",
    "axis",
    "picture",
    "thebibliography",
}

TEXT_COMMANDS = {
    "chapter",
    "section",
    "subsection",
    "subsubsection",
    "caption",
    "textbf",
    "textit",
    "emph",
}


def protect(text: str):
    tokens = {}

    def add(match):
        key = f"ZXQ{len(tokens):04d}QXZ"
        tokens[key] = match.group(0)
        return key

    patterns = [
        r"\\\[[\s\S]*?\\\]",
        r"\\\([\s\S]*?\\\)",
        r"\$\$[\s\S]*?\$\$",
        r"\$[^$]*\$",
        r"\\(?:cite|citep|citet|ref|eqref|label|url|href|includegraphics|input|ldots|dots|times|frac|sqrt|sum|prod|log|min|max|argmin|argmax|Pr|mathbb|mathbf|mathrm|mathcal|mathsf|hat|tilde|bar|overline|underline|left|right|big|Big|bigg|Bigg|leq|geq|neq|approx|sim|infty|alpha|beta|gamma|delta|epsilon|varepsilon|theta|lambda|mu|sigma|phi|varphi|omega|Omega|Delta|Pi|Lambda|Theta|rho|eta|ell|cdot|circ|oplus|otimes|rightarrow|leftarrow|leftrightarrow|to|in|notin|subset|supset|cup|cap|emptyset|forall|exists|times|pm)(?:\*?)(?:\[[^\]]*\])?(?:\{[^{}]*\})*",
        r"\\[a-zA-Z]+(?:\*?)(?:\[[^\]]*\])?",
        r"\\\\",
        r"&",
        r"%",
        r"~",
        r"--",
        r"---",
    ]
    for pattern in patterns:
        text = re.sub(pattern, add, text)
    return text, tokens


def unprotect(text: str, tokens: dict):
    for key, value in tokens.items():
        text = text.replace(key, value)
        text = text.replace(key.lower(), value)
    return text


def translate_text(text: str, translator, cache: dict) -> str:
    if not text.strip():
        return text
    stripped = text.strip()
    if not re.search(r"[A-Za-z]", stripped):
        return text
    if stripped in cache:
        translated = cache[stripped]
    else:
        protected, tokens = protect(stripped)
        try:
            translated = translator.translate(protected)
        except Exception:
            time.sleep(2)
            translated = translator.translate(protected)
        translated = unprotect(translated, tokens)
        cache[stripped] = translated
        time.sleep(0.08)
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    return leading + translated + trailing


def translate_braced_commands(line: str, translator, cache: dict) -> str:
    for command in TEXT_COMMANDS:
        pattern = re.compile(rf"(\\{command}\*?(?:\[[^\]]*\])?\{{)([^{{}}]*)(\}})")

        def repl(match):
            return match.group(1) + translate_text(match.group(2), translator, cache) + match.group(3)

        line = pattern.sub(repl, line)
    return line


def should_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("%"):
        return True
    if any(stripped.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    if re.fullmatch(r"\\(toprule|midrule|bottomrule|hline|centering|small|normalsize|large|Large|bfseries|itshape|normalfont|noindent|par|item)\b.*", stripped):
        return True
    return False


def translate_file(src: Path, dst: Path, translator, cache: dict):
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    out_lines = []
    raw_env = None
    math_env = None
    paragraph = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            text = "".join(paragraph)
            out_lines.append(translate_text(text, translator, cache))
            paragraph = []

    for line in lines:
        begin = re.search(r"\\begin\{([^}]+)\}", line)
        end = re.search(r"\\end\{([^}]+)\}", line)

        if raw_env:
            out_lines.append(line)
            if end and end.group(1) == raw_env:
                raw_env = None
            continue

        if math_env:
            out_lines.append(line)
            if end and end.group(1) == math_env:
                math_env = None
            continue

        if begin and begin.group(1) in RAW_ENVS:
            flush_paragraph()
            raw_env = begin.group(1)
            out_lines.append(line)
            continue

        if begin and begin.group(1) in MATH_ENVS:
            flush_paragraph()
            math_env = begin.group(1)
            out_lines.append(line)
            continue

        if should_skip_line(line):
            flush_paragraph()
            out_lines.append(line)
            continue

        if re.search(r"\\(" + "|".join(TEXT_COMMANDS) + r")\*?(?:\[[^\]]*\])?\{", line):
            flush_paragraph()
            out_lines.append(translate_braced_commands(line, translator, cache))
            continue

        if line.strip() == "":
            flush_paragraph()
            out_lines.append(line)
            continue

        paragraph.append(line)

    flush_paragraph()
    dst.write_text("".join(out_lines), encoding="utf-8")


def patch_vietnamese_main(path: Path):
    text = path.read_text(encoding="utf-8")
    replacements = {
        r"\renewcommand{\contentsname}{TABLE OF CONTENTS}": r"\renewcommand{\contentsname}{MỤC LỤC}",
        r"\renewcommand{\listfigurename}{LIST OF FIGURES}": r"\renewcommand{\listfigurename}{DANH MỤC HÌNH VẼ}",
        r"\renewcommand{\listtablename}{LIST OF TABLES}": r"\renewcommand{\listtablename}{DANH MỤC BẢNG}",
        r"\renewcommand{\bibname}{REFERENCES}": r"\renewcommand{\bibname}{TÀI LIỆU THAM KHẢO}",
        r"\renewcommand{\figurename}{Figure}": r"\renewcommand{\figurename}{Hình}",
        r"\renewcommand{\tablename}{Table}": r"\renewcommand{\tablename}{Bảng}",
        r"\renewcommand{\chaptername}{Chapter}": r"\renewcommand{\chaptername}{Chương}",
        r"\newcommand{\thesisTitle}{UTILIZATION OF ARTIFICIAL NEURAL NETWORKS IN LDPC DECODING FOR BIBCM-ID SYSTEMS}": r"\newcommand{\thesisTitle}{ỨNG DỤNG MẠNG NƠ-RON NHÂN TẠO TRONG GIẢI MÃ LDPC CHO HỆ THỐNG BIBCM-ID}",
        r"\newcommand{\fieldName}{Telecommunications Engineering}": r"\newcommand{\fieldName}{Kỹ thuật viễn thông}",
        r"\newcommand{\majorName}{Telecommunications Engineering}": r"\newcommand{\majorName}{Kỹ thuật viễn thông}",
        r"\newcommand{\academyName}{MILITARY TECHNICAL ACADEMY}": r"\newcommand{\academyName}{HỌC VIỆN KỸ THUẬT QUÂN SỰ}",
        "MINISTRY OF EDUCATION AND TRAINING": "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
        "MINISTRY OF DEFENCE": "BỘ QUỐC PHÒNG",
        "MASTER'S THESIS": "LUẬN VĂN THẠC SĨ",
        "Field:": "Ngành:",
        "Major:": "Chuyên ngành:",
        "Code:": "Mã số:",
        "SCIENTIFIC SUPERVISOR": "NGƯỜI HƯỚNG DẪN KHOA HỌC",
        "First supervisor:": "Người hướng dẫn thứ nhất:",
        "Second supervisor:": "Người hướng dẫn thứ hai:",
        "Hanoi -- 2026": "Hà Nội -- 2026",
        "DECLARATION": "LỜI CAM ĐOAN",
        "THESIS AUTHOR": "TÁC GIẢ LUẬN VĂN",
        "TABLE OF CONTENTS": "MỤC LỤC",
        "MASTER THESIS SUMMARY": "TÓM TẮT LUẬN VĂN THẠC SĨ",
        "LIST OF ABBREVIATIONS": "DANH MỤC CHỮ VIẾT TẮT",
        "LIST OF MATHEMATICAL SYMBOLS": "DANH MỤC KÝ HIỆU TOÁN HỌC",
        "LIST OF TABLES": "DANH MỤC BẢNG",
        "LIST OF FIGURES": "DANH MỤC HÌNH VẼ",
        "INTRODUCTION": "MỞ ĐẦU",
        "REFERENCES": "TÀI LIỆU THAM KHẢO",
        "SCIENTIFIC PUBLICATIONS": "CÁC CÔNG TRÌNH KHOA HỌC ĐÃ CÔNG BỐ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace(r"\renewcommand{\cftchappresnum}{Chapter\ }", r"\renewcommand{\cftchappresnum}{Chương\ }")
    text = text.replace(r"\addcontentsline{toc}{chapter}{Declaration}", r"\addcontentsline{toc}{chapter}{LỜI CAM ĐOAN}")
    text = text.replace(r"\manualtocline{Declaration}{i}", r"\manualtocline{LỜI CAM ĐOAN}{i}")
    text = text.replace(r"\input{body_from_docx.tex}", r"\input{body_from_docx_vi.tex}")
    path.write_text(text, encoding="utf-8")


def patch_inputs(path: Path):
    text = path.read_text(encoding="utf-8")
    for name in [
        "expanded_ch1",
        "expanded_ch1_ml",
        "expanded_ch2",
        "expanded_ch3",
        "expanded_demapper",
        "expanded_validation",
    ]:
        text = text.replace(f"\\input{{{name}.tex}}", f"\\input{{{name}_vi.tex}}")
    path.write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    if (ROOT / "figures").exists():
        shutil.copytree(ROOT / "figures", OUT / "figures", dirs_exist_ok=True)

    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    else:
        cache = {}
    translator = GoogleTranslator(source="en", target="vi")

    for file_name in FILES:
        src = ROOT / file_name
        dst_name = file_name.replace(".tex", "_vi.tex")
        if file_name == "thesis.tex":
            dst_name = "thesis_vi.tex"
        dst = OUT / dst_name
        print(f"Translating {file_name} -> {dst.name}")
        translate_file(src, dst, translator, cache)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    patch_vietnamese_main(OUT / "thesis_vi.tex")
    patch_inputs(OUT / "body_from_docx_vi.tex")
    print("Done.")


if __name__ == "__main__":
    main()
