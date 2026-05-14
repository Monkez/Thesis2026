from pathlib import Path
import re
import sys

from docx import Document


ROOT = Path(__file__).resolve().parent
DOCX = ROOT.parent / "LuanVan_3Chapters_DeepReviewed.docx"
OUT = ROOT / "body_from_docx.tex"


EQUATIONS = {
    "(1.1)": r"""\begin{equation}
\mathbf{u}\xrightarrow{\mathrm{Encoder}}\mathbf{c}\xrightarrow{\pi}\mathbf{v}\xrightarrow{\mu}\mathbf{x}\xrightarrow{\mathrm{Channel}}\mathbf{y}.
\end{equation}""",
    "(1.2)": r"""\begin{equation}
R_c=\frac{k}{n},\qquad R_b=R_c\log_2(M)\ \mathrm{bit/symbol}.
\end{equation}""",
    "(1.3)": r"""\begin{equation}
\frac{E_b}{N_0}=\frac{E_s}{N_0}-10\log_{10}\!\left(R_c\log_2 M\right)\ \mathrm{dB}.
\end{equation}""",
    "(1.4)": r"""\begin{equation}
L(b_k|y)=\ln\frac{P(b_k=1|y)}{P(b_k=0|y)}.
\end{equation}""",
    "(1.5)": r"""\begin{equation}
L_{\mathrm{E}}(b_k)=L_{\mathrm{APP}}(b_k)-L_{\mathrm{A}}(b_k)-L_{\mathrm{ch}}(b_k).
\end{equation}""",
    "(1.6)": r"""\begin{equation}
\mathbf{L}^{(t)}_{\mathrm{dec}}=
\mathcal{D}\!\left(\mathbf{L}_{\mathrm{ch}},\Pi^{-1}\mathbf{L}^{(t)}_{\mathrm{dem}}\right).
\end{equation}""",
    "(1.7)": r"""\begin{equation}
\mathbf{L}^{(t+1)}_{\mathrm{dem}}=
\mathcal{M}\!\left(\mathbf{y},\Pi\mathbf{L}^{(t)}_{\mathrm{dec}}\right).
\end{equation}""",
    "(1.8)": r"""\begin{equation}
\mathrm{BER}=\frac{N_{\mathrm{err}}}{N_{\mathrm{bit}}}.
\end{equation}""",
    "(2.1)": r"""\begin{equation}
\mathbf{H}\mathbf{c}^{T}=\mathbf{0}.
\end{equation}""",
    "(2.2)": r"""\begin{equation}
L_{\mathrm{ch},j}=\ln\frac{p(y_j|c_j=1)}{p(y_j|c_j=0)}.
\end{equation}""",
    "(2.3)": r"""\begin{equation}
L_{j\rightarrow i}^{(t)}=L_{\mathrm{ch},j}+\sum_{i'\in\mathcal{N}(j)\setminus i}L_{i'\rightarrow j}^{(t-1)}.
\end{equation}""",
    "(2.4)": r"""\begin{equation}
L_{i\rightarrow j}^{(t)}=
2\tanh^{-1}\!\prod_{j'\in\mathcal{N}(i)\setminus j}\tanh\!\left(\frac{L_{j'\rightarrow i}^{(t)}}{2}\right).
\end{equation}""",
    "(2.5)": r"""\begin{equation}
L_{i\rightarrow j}^{(t)}\approx
\left(\prod_{j'\ne j}\mathrm{sign}(L_{j'\rightarrow i}^{(t)})\right)
\min_{j'\ne j}|L_{j'\rightarrow i}^{(t)}|.
\end{equation}""",
    "(2.6)": r"""\begin{equation}
L_{i\rightarrow j,\mathrm{NMS}}^{(t)}=\alpha L_{i\rightarrow j,\mathrm{MS}}^{(t)},\qquad 0<\alpha\le 1.
\end{equation}""",
    "(2.7)": r"""\begin{equation}
L_{i\rightarrow j,\mathrm{OMS}}^{(t)}=\mathrm{sign}(L_{\mathrm{MS}})\max(|L_{\mathrm{MS}}|-\beta,0).
\end{equation}""",
    "(2.8)": r"""\begin{equation}
L_{\mathrm{APP},j}^{(t)}=L_{\mathrm{ch},j}+\sum_{i\in\mathcal{N}(j)}L_{i\rightarrow j}^{(t)}.
\end{equation}""",
    "(2.9)": r"""\begin{equation}
\hat{c}_j=\begin{cases}1,&L_{\mathrm{APP},j}\ge 0,\\0,&L_{\mathrm{APP},j}<0.\end{cases}
\end{equation}""",
    "(2.10)": r"""\begin{equation}
\mathbf{s}=\mathbf{H}\hat{\mathbf{c}}^{T}.
\end{equation}""",
    "(2.11)": r"""\begin{equation}
\hat{\mathbf{r}}=f_{\theta}(\mathbf{q}).
\end{equation}""",
    "(2.12)": r"""\begin{equation}
\mathrm{ReLU}(x)=\max(x,0).
\end{equation}""",
    "(2.13)": r"""\begin{equation}
N_{\theta}=d_{\mathrm{in}}d_h+d_h+d_hd_{\mathrm{out}}+d_{\mathrm{out}}.
\end{equation}""",
    "(2.14)": r"""\begin{equation}
\mathcal{L}_{\mathrm{MAE}}=\frac{1}{N}\sum_{n=1}^{N}\left|\hat{r}_n-r_n\right|.
\end{equation}""",
    "(2.15)": r"""\begin{equation}
r_n=\begin{cases}r_n^{\mathrm{SPA}},&\mathrm{SPA\ label},\\ \eta_n r_n^{\mathrm{SPA}},&\mathrm{reliability\ aware\ label}.\end{cases}
\end{equation}""",
    "(2.16)": r"""\begin{equation}
\theta^{\star}=\arg\min_{\theta}\mathbb{E}_{\mathbf{q},\mathbf{r}}\left[\ell(f_{\theta}(\mathbf{q}),\mathbf{r})\right].
\end{equation}""",
    "(2.17)": r"""\begin{equation}
\theta_{t+1}=\theta_t-\gamma_t\nabla_{\theta}\mathcal{L}(\theta_t).
\end{equation}""",
    "(2.18)": r"""\begin{equation}
\mathrm{SNR}_{\mathrm{dB}}=10\log_{10}\frac{E_s}{N_0}.
\end{equation}""",
    "(3.1)": r"""\begin{equation}
y=hx+n.
\end{equation}""",
    "(3.2)": r"""\begin{equation}
L(b_k|y)=\ln\frac{\sum_{s\in\mathcal{S}_{k,1}}\exp(-|y-hs|^2/N_0)}
{\sum_{s\in\mathcal{S}_{k,0}}\exp(-|y-hs|^2/N_0)}.
\end{equation}""",
    "(3.3)": r"""\begin{equation}
\mathrm{PAPR}=\frac{\max_n |x[n]|^2}{\mathbb{E}\{|x[n]|^2\}}.
\end{equation}""",
    "(3.4)": r"""\begin{equation}
d_{\min}=\min_{i\ne j}\|s_i-s_j\|_2.
\end{equation}""",
    "(3.5)": r"""\begin{equation}
m\rightarrow x=f_{\theta}(m),\qquad y=\mathcal{H}(x),\qquad \hat{p}(m|y)=g_{\phi}(y).
\end{equation}""",
    "(3.6)": r"""\begin{equation}
\mathbb{E}\{|x|^2\}\le P_0.
\end{equation}""",
    "(3.7)": r"""\begin{equation}
x_{\mathrm{norm}}=\frac{x_{\mathrm{raw}}}{\sqrt{\mathbb{E}\{|x_{\mathrm{raw}}|^2\}+\epsilon}}.
\end{equation}""",
    "(3.8)": r"""\begin{equation}
\mathcal{L}_{\mathrm{CE}}=-\sum_{m=1}^{M}\mathbf{1}\{m\}\log\hat{p}(m|y).
\end{equation}""",
    "(3.9)": r"""\begin{equation}
\mathrm{SER}=\frac{N_{\mathrm{sym,err}}}{N_{\mathrm{sym}}}.
\end{equation}""",
    "(3.10)": r"""\begin{equation}
g(r)=\frac{r}{\left[1+\left(r/A_{\mathrm{sat}}\right)^{2p}\right]^{1/(2p)}}.
\end{equation}""",
    "(3.11)": r"""\begin{equation}
\Phi(r)=\frac{\alpha_{\mathrm{PM}}r^2}{1+\beta_{\mathrm{PM}}r^2}.
\end{equation}""",
    "(3.12)": r"""\begin{equation}
y_{\mathrm{CFO}}[k]=x[k]e^{j2\pi \Delta f T_s k}+w[k].
\end{equation}""",
    "(3.13)": r"""\begin{equation}
y[n]=\sum_{\ell=0}^{L-1}h[\ell]x[n-\ell]+w[n].
\end{equation}""",
    "(3.14)": r"""\begin{equation}
h[\ell]\sim\mathcal{CN}(0,\sigma_{\ell}^{2}),\qquad \sum_{\ell}\sigma_{\ell}^{2}=1.
\end{equation}""",
    "(3.15)": r"""\begin{equation}
\mathcal{L}_{\mathrm{BCE}}=-\frac{1}{K}\sum_{k=1}^{K}\left[b_k\log \hat{b}_k+(1-b_k)\log(1-\hat{b}_k)\right].
\end{equation}""",
}


FIGURES_AFTER_HEADING = {
    "1.2. Modulation, Channel Coding, and Interleaving": r"""\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{sys_diagram.png}
\caption{End-to-end transceiver structure used as a reference for the BIBCM-ID discussion.}
\end{figure}""",
    "3.2. Autoencoder Communication Model": r"""\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{autoencoder.png}
\caption{AutoEncoder communication model for learned modulation.}
\end{figure}""",
    "3.3. Non-Ideal Channel Conditions": r"""\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{pa_characteristic.png}
\caption{Power-amplifier nonlinearity used in the non-ideal channel analysis.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{cfo_effect.png}
\caption{Carrier-frequency-offset effect on constellation evolution over time.}
\end{figure}""",
    "3.4. Available Simulation Scenarios": r"""\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{ae_scenario1_ber.png}
\caption{BER result for the single-symbol AutoEncoder scenario under PA/Rayleigh conditions.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{ae_scenario2_cfo0005.png}
\caption{BER result for the multi-symbol AutoEncoder scenario with CFO = 0.005.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{ae_scenario2_cfo001.png}
\caption{BER result for the multi-symbol AutoEncoder scenario with CFO = 0.01.}
\end{figure}
\begin{figure}[H]
\centering
\includegraphics[width=0.86\textwidth]{ae_scenario3_fec.png}
\caption{BER result for the AutoEncoder scenario with neural FEC characteristics.}
\end{figure}""",
}


def esc(text: str) -> str:
    text = text.replace("\u00a0", " ")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def strip_number(title: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.\s*", "", title).strip()


def chapter_title(title: str) -> str:
    m = re.match(r"^CHAPTER\s+\d+\.\s*(.*)$", title, flags=re.I)
    return m.group(1).strip() if m else title.strip()


def main():
    if "--force" not in sys.argv:
        raise SystemExit(
            "This script regenerates body_from_docx.tex from the older DOCX source. "
            "The curated LaTeX thesis has been manually reviewed; run with --force "
            "only if you intentionally want to overwrite it."
        )

    doc = Document(DOCX)
    lines = []
    lines.append(r"\chapter*{INTRODUCTION}")
    lines.append(r"\addcontentsline{toc}{chapter}{\textbf{INTRODUCTION}}")
    lines.append(
        "This thesis studies the use of artificial neural networks in two components related to BIBCM-ID systems: LDPC channel decoding and modulation. "
        "The interleaver is treated as an important and well-studied component of the BIBCM-ID chain, while the thesis focuses on the two remaining directions where the available materials provide direct evidence: neural assistance for LDPC decoding and AutoEncoder-based modulation under non-ideal wireless channels."
    )
    lines.append("")
    lines.append(
        "The objective is to analyze how deep learning can improve or approximate selected processing blocks while preserving the communication-theory structure of BIBCM-ID. "
        "The thesis therefore follows a component-level methodology: the decoder part is studied through ANN-assisted LDPC message updates, and the modulation part is studied through learned AutoEncoder constellations and sequence processing. "
        "This scope is consistent with the available simulation evidence and avoids over-extending the conclusions beyond the implemented results."
    )
    lines.append("")

    in_body = False
    in_refs = False
    ref_idx = 1
    in_bib = False
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name
        if text.startswith("CHAPTER 1."):
            in_body = True
        if not in_body:
            continue

        if style == "Heading 1":
            if text == "REFERENCES":
                lines.append(r"\begin{thebibliography}{99}")
                lines.append(r"\addcontentsline{toc}{chapter}{\textbf{REFERENCES}}")
                in_refs = True
                in_bib = True
                continue
            if in_bib:
                lines.append(r"\end{thebibliography}")
                in_bib = False
            if text == "APPENDICES":
                lines.append(r"\appendix")
                lines.append(r"\chapter*{APPENDICES}")
                lines.append(r"\addcontentsline{toc}{chapter}{\textbf{APPENDICES}}")
                in_refs = False
                continue
            if text == "CONCLUSION AND RECOMMENDATIONS":
                lines.append(r"\chapter*{CONCLUSION AND RECOMMENDATIONS}")
                lines.append(r"\addcontentsline{toc}{chapter}{\textbf{CONCLUSION AND RECOMMENDATIONS}}")
                in_refs = False
                continue
            title = chapter_title(text).upper()
            lines.append(rf"\chapter[\textbf{{{esc(title)}}}]{{{esc(title)}}}")
            in_refs = False
            continue

        if in_refs:
            item = re.sub(r"^\[\d+\]\s*", "", text)
            lines.append(rf"\bibitem{{docxref{ref_idx}}} {esc(item)}")
            ref_idx += 1
            continue

        if style == "Heading 2":
            title = strip_number(text)
            lines.append(rf"\section{{{esc(title)}}}")
            if text in FIGURES_AFTER_HEADING:
                lines.append(FIGURES_AFTER_HEADING[text])
        elif style == "Heading 3":
            title = strip_number(text)
            lines.append(rf"\subsection{{{esc(title)}}}")
        elif style == "List Bullet":
            lines.append(r"\begin{itemize}")
            lines.append(rf"\item {esc(text)}")
            lines.append(r"\end{itemize}")
        elif text in EQUATIONS:
            lines.append(EQUATIONS[text])
        else:
            lines.append(esc(text))
            lines.append("")

    if in_bib:
        lines.append(r"\end{thebibliography}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote body_from_docx.tex")


if __name__ == "__main__":
    main()
