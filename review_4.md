# Review 4 - Nhận xét phản biện chi tiết luận văn

**Đối tượng rà soát:** `latex/thesis.pdf` và các nguồn LaTeX hiện tại trong thư mục `latex/`  
**Thời điểm rà soát:** 15/05/2026  
**Vai trò rà soát:** phản biện luận văn thạc sĩ, tập trung vào lỗi trình bày, nội dung, học thuật, công thức toán học và độ chặt của lập luận.

## 1. Nhận xét tổng quan

Bản luận văn mới đã cải thiện rõ so với các bản rà soát trước. Chương 2 hiện có cấu hình mô phỏng, hình huấn luyện, hình BER, bảng so sánh bộ giải mã và phần diễn giải thận trọng hơn về mức tăng 0.2--0.4 dB. Các lỗi nghiêm trọng đã từng nêu như sai số chiều đầu vào `d_in`, công thức đếm tham số MLP, chú thích hình còn chữ "redrawn", hoặc cách trình bày NMS/OMS như đường cong mô phỏng đều đã được chỉnh theo hướng tốt hơn.

Tuy nhiên, nếu đọc ở vai trò phản biện học thuật, bản hiện tại vẫn còn một số điểm có thể bị hỏi trong buổi bảo vệ. Các vấn đề chính không nằm ở bố cục tổng thể, mà nằm ở mức độ tái lập thí nghiệm, độ chính xác của quy ước LLR/BPSK, định nghĩa nhãn reliability-aware, và sự liên kết giữa bối cảnh BIBCM-ID với phần đóng góp chính ANN-LDPC. Nói cách khác, luận văn đã có hình hài tốt hơn, nhưng một số khẳng định kỹ thuật vẫn cần được khóa chặt để tránh bị xem là mô tả định tính nhiều hơn chứng minh định lượng.

## 2. Các vấn đề ưu tiên cao cần sửa

### 2.1. Cần định nghĩa rõ quy ước LLR và ánh xạ BPSK

Trong Chương 1, luận văn định nghĩa:

```tex
L(b_k|y)=\ln\frac{P(b_k=1|y)}{P(b_k=0|y)}.
```

Trong Chương 2, hard decision được viết:

```tex
\hat{c}_j=1 nếu L_{\mathrm{APP},j}\ge 0,
\hat{c}_j=0 nếu L_{\mathrm{APP},j}<0.
```

Về logic nội bộ, hai biểu thức này là nhất quán nếu luận văn dùng quy ước `LLR = log P(1)/P(0)`. Tuy nhiên, trong nhiều tài liệu mã kênh và BPSK, người đọc quen với ánh xạ `0 -> +1`, `1 -> -1`, khi đó LLR kênh thường được viết theo quy ước ngược dấu, tức `log P(0|y)/P(1|y)` hoặc một dạng tỉ lệ với `+2y/sigma^2` cho bit 0. Vì vậy, nếu luận văn không nêu ánh xạ BPSK cụ thể, phản biện có thể nghi ngờ dấu của LLR kênh và hard decision.

**Vị trí liên quan:** `latex/body_from_docx.tex`, phần định nghĩa LLR Chương 1 và các phương trình (2.2), (2.9).

**Đề xuất sửa:** thêm ngay sau phương trình LLR hoặc ở đầu Chương 2 một câu kiểu:

> In this thesis, all LLRs follow the convention \(L=\log P(b=1|y)/P(b=0|y)\). Therefore, a positive LLR favors bit 1. The BPSK mapping and the channel LLR sign are chosen consistently with this convention.

Nếu ánh xạ thực tế trong mô phỏng là `0 -> +1`, `1 -> -1`, cần viết rõ công thức LLR kênh theo dấu tương ứng, tránh chỉ nói "proportional to the received sample" vì câu này chưa đủ xác định dấu.

### 2.2. Cấu hình mô phỏng BER chưa đủ để tái lập

Bảng 2.1 đã tốt hơn vì có nêu QC-LDPC `(332,664)`, BPSK/AWGN, Adam, MAE, batch size, epoch và dải SNR. Tuy nhiên, một phản biện vẫn chưa thể tái lập Hình 2.3 vì thiếu các thông tin sau:

- Số frame hoặc số bit mô phỏng tại từng điểm SNR.
- Số lỗi tối thiểu dùng để dừng mô phỏng tại từng SNR.
- Số iteration tối đa cụ thể của decoder.
- Có dùng syndrome stopping hay không trong đường BER đang vẽ.
- Training SNR và testing SNR có trùng nhau hay không.
- Seed ngẫu nhiên hoặc cách chia tập train/validation/test.
- Nguồn hoặc cấu trúc cụ thể của ma trận QC-LDPC.

Bản hiện tại có câu tự giới hạn rằng đường cong chỉ là "trend-level evidence". Câu này là đúng và nên giữ. Nhưng nếu luận văn vẫn nhắc đến mức tăng gần `10^{-5}`, phản biện có quyền hỏi số lượng lỗi quan sát được tại vùng BER thấp. Không có error-count log thì mức tăng 0.2--0.4 dB chỉ nên xem là xu hướng, không nên xem là kết luận định lượng mạnh.

**Vị trí liên quan:** `latex/body_from_docx.tex`, Bảng 2.1 và đoạn Performance Discussion.

**Đề xuất sửa:** thêm một bảng nhỏ hoặc một đoạn "Simulation reproducibility details" sau Bảng 2.1. Nếu chưa có dữ liệu thật, nên viết rõ:

> The present BER curve is used to compare algorithmic trends. A standard low-BER claim would require a longer simulation with per-SNR frame counts and error-event logs.

Nếu có dữ liệu thật, nên thêm bảng gồm `SNR`, `number of frames`, `number of bits`, `observed bit errors`, `BER`.

### 2.3. "Fixed maximum iteration count" nhưng chưa nêu giá trị

Trong Bảng 2.1, luận văn ghi "Fixed maximum iteration count", nhưng không nêu `I_max` bằng bao nhiêu. Đây là thiếu sót quan trọng vì BER và độ phức tạp của LDPC decoder phụ thuộc trực tiếp vào số vòng lặp. Phần complexity sau đó lại dùng công thức:

```tex
C_total ≈ E I_max C_MLP.
```

Như vậy `I_max` vừa là tham số mô phỏng, vừa là tham số đánh giá độ phức tạp. Nếu không nêu giá trị, người đọc không kiểm tra được tính công bằng giữa SPA, Min-Sum và ANN update.

**Đề xuất sửa:** thay dòng trong Bảng 2.1 thành dạng cụ thể:

> Maximum iterations: \(I_{\max}=...\). All compared decoders use the same maximum iteration count. Syndrome stopping is disabled/enabled as follows: ...

Nếu có dùng early stopping, cần nói rõ BER curve dùng average iteration hay maximum iteration.

### 2.4. Nhãn reliability-aware cần định nghĩa chặt hơn để tránh bị hiểu là dùng oracle

Luận văn viết rằng nếu dấu của noisy SPA message khác với "reference message", độ lớn nhãn sẽ bị giảm. Đây là ý tưởng hợp lý, nhưng "reference message" hiện chưa được định nghĩa đủ chặt. Phản biện có thể hỏi:

- Reference message lấy từ đâu?
- Nó có dùng thông tin codeword thật không?
- Nó có dùng SPA ở SNR cao, clean message, hoặc một decoder mạnh hơn không?
- Khi triển khai thực tế, reference này có tồn tại không?
- Nếu chỉ tồn tại khi huấn luyện, liệu mô hình có học từ thông tin oracle không?

Luận văn đã có một đoạn tự cảnh báo về rủi ro oracle trong `expanded_ch2.tex`, đây là điểm tốt. Tuy nhiên, phần mô tả chính trong Chương 2 vẫn cần nêu rõ cơ chế tạo nhãn.

**Vị trí liên quan:** `latex/body_from_docx.tex`, mục Training Data and Labeling Strategy; `latex/expanded_ch2.tex`, đoạn về oracle risk.

**Đề xuất sửa:** thêm một đoạn sau phương trình label:

> The reference message used in label construction is obtained from ... . It is used only during offline training and is not required during decoder inference. Therefore, the deployed ANN decoder receives only the local incoming messages available to a conventional check-node update.

Nếu reference thật sự dùng codeword hoặc thông tin không có ở receiver, phải nói thẳng đây là "genie-aided label construction for feasibility analysis", không nên trình bày như một thuật toán triển khai hoàn chỉnh.

### 2.5. Đầu vào MLP bốn chiều chưa được đặc tả đủ cho QC-LDPC thực tế

Luận văn đã sửa đúng `d_in=4`, `d_h=8`, `d_out=1`, 49 tham số và 40 MAC. Tuy nhiên, câu "four input values" vẫn chưa đủ rõ. Với check-node update, số message đầu vào phụ thuộc vào bậc check node. Nếu ma trận QC-LDPC có check degree khác 5, hoặc có nhiều bậc check khác nhau, MLP bốn đầu vào không thể trực tiếp nhận toàn bộ các message vào trừ khi có quy tắc chọn đặc trưng.

Luận văn có nói rằng nếu check node có bậc khác thì có thể dùng degree-specific network, chọn bốn đặc trưng reliability, hoặc pad/truncate. Đây là hướng đúng, nhưng vẫn còn hơi mở. Một phản biện có thể hỏi: trong kết quả Hình 2.3, phương án nào đã được dùng?

**Đề xuất sửa:** nêu cụ thể vector đầu vào:

```tex
q = [ ... ] \in R^4
```

Ví dụ:

- Nếu check degree là 5 và đang tính message ra một cạnh, bốn đầu vào chính là bốn VN-to-CN messages còn lại.
- Nếu check degree lớn hơn, phải nói bốn đầu vào là `four smallest magnitudes`, `sign product`, hoặc đặc trưng nào khác.
- Nếu dùng pad/truncate, cần nói thứ tự chọn và ảnh hưởng tới tính đối xứng của check-node update.

Đây là điểm quan trọng vì nó liên quan trực tiếp đến tính đúng của việc thay thế check-node update.

## 3. Vấn đề học thuật và độ chặt của lập luận

### 3.1. Mối liên kết giữa BIBCM-ID và kết quả ANN-LDPC vẫn cần được nhấn chặt hơn

Tên và phần mở đầu của luận văn đặt trong bối cảnh BIBCM-ID. Tuy nhiên, kết quả chính ở Chương 2 hiện được đánh giá trên BPSK/AWGN để cô lập check-node update. Cách làm này hợp lý nếu được trình bày như một bước validation có kiểm soát. Luận văn đã có đoạn nói BPSK/AWGN là controlled first evaluation, nhưng cần nhấn mạnh hơn ở phần cuối Chương 1 và đầu Chương 2 để tránh cảm giác "tên luận văn nói BIBCM-ID nhưng thí nghiệm chính lại là BPSK/AWGN".

**Đề xuất sửa:** thêm một đoạn chuyển tiếp rõ:

> Although the target receiver context is BIBCM-ID, the first experimental validation uses BPSK/AWGN LLRs to isolate the LDPC check-node approximation. This controlled setting avoids mixing demodulator mismatch, interleaver effects, and LDPC graph behavior. Integration with high-order BIBCM-ID soft demodulation is treated as the next validation stage.

Đoạn này sẽ làm flow hợp lý hơn và bảo vệ được phạm vi đóng góp.

### 3.2. Thuật ngữ BIBCM-ID cần có định nghĩa và nguồn tham chiếu mạnh hơn

Luận văn dùng thuật ngữ BIBCM-ID, trong khi nhiều tài liệu phổ biến dùng BICM-ID. Nếu BIBCM-ID là thuật ngữ riêng của hướng nghiên cứu hoặc nhấn mạnh "Block-Interleaved Bit-Interleaved Coded Modulation with Iterative Decoding", cần định nghĩa rõ ngay từ đầu và phân biệt với BICM-ID chuẩn.

**Đề xuất sửa:** ở Chương 1, sau lần xuất hiện đầu tiên của BIBCM-ID, nên có một đoạn:

> BIBCM-ID is used in this thesis to emphasize ... . It is closely related to BICM-ID, but the block-interleaving structure is treated explicitly because ...

Nếu không có nguồn riêng cho BIBCM-ID, nên dùng cách viết thận trọng: "BICM-ID/BIBCM-ID-like receiver" hoặc "block-interleaved BICM-ID receiver".

### 3.3. Một số khẳng định về interleaver còn thiên về mô tả, thiếu dẫn chứng chuyên biệt

Luận văn nói interleaver làm giảm phụ thuộc thống kê và có vai trò quan trọng trong iterative decoding. Đây là đúng về nguyên lý. Tuy nhiên, nếu luận văn nhắc đến các hướng như algebraic interleaver, protograph-based interleaver, contention-free interleaver, hoặc ảnh hưởng của interleaver tới convergence, nên có nguồn tham chiếu cụ thể hơn.

**Đề xuất sửa:** hoặc bổ sung citation cho interleaver design trong BICM-ID/LDPC, hoặc giảm độ mạnh của câu khẳng định. Vì luận văn không đề xuất interleaver mới, cách tốt nhất là viết theo hướng:

> Interleaver design is an important but separate problem. This thesis does not optimize the interleaver; it treats the interleaver as a fixed component and focuses on the neural processing blocks.

### 3.4. Kết luận Chương 2 đã thận trọng hơn nhưng vẫn có câu "improves clearly over Min-Sum"

Câu này nhìn chung chấp nhận được vì Hình 2.3 có so với Min-Sum. Tuy nhiên, nếu muốn chuẩn phản biện chặt hơn, nên gắn câu đó với đúng điều kiện mô phỏng:

> under the evaluated QC-LDPC code, BPSK/AWGN channel, SNR grid, and iteration setting.

Không nên để người đọc hiểu đây là kết luận tổng quát cho mọi LDPC code hoặc mọi kênh.

## 4. Vấn đề công thức toán học và ký hiệu

### 4.1. Cần dùng `\label` và `\ref` thay cho đánh số thủ công nhiều hơn

Một số phương trình Chương 1 đã dùng `\label` và `\eqref`, nhưng nhiều phương trình Chương 2 và Chương 3 vẫn được gọi bằng "Equation (2.x)" trong văn bản. Cách này dễ sai khi thêm/bớt phương trình, đặc biệt luận văn đang còn chỉnh sửa.

**Đề xuất sửa:** gán `\label{...}` cho các phương trình quan trọng:

- parity-check condition;
- channel LLR;
- VN update;
- SPA CN update;
- Min-Sum/NMS/OMS;
- hard decision;
- MAE loss;
- reliability-aware label;
- parameter count;
- complexity count;
- Rapp PA model;
- CFO model.

Sau đó dùng `Equation~\eqref{...}`. Đây là sửa trình bày nhưng cũng nâng độ chuyên nghiệp học thuật.

### 4.2. Công thức Rapp PA model cần kiểm tra trực quan trong PDF

Trong nguồn LaTeX, mô hình Rapp được viết đúng dạng phân số:

```tex
g(r)=\frac{r}{\left[1+\left(r/A_{\mathrm{sat}}\right)^{2p}\right]^{1/(2p)}}.
```

Đây là dạng hợp lý vì biên độ bị nén khi `r` tăng. Tuy nhiên, khi trích xuất text từ PDF, công thức có thể bị mất cấu trúc phân số, làm người đọc tưởng thành phép nhân:

```text
g(r) = r [1 + ...]^{1/(2p)}
```

Nếu PDF hiển thị đúng fraction bar thì không có lỗi. Nhưng cần kiểm tra trực quan vì nếu công thức in ra thật sự mất mẫu số, ý nghĩa vật lý bị đảo ngược: thay vì nén biên độ, mô hình sẽ khuếch đại mạnh hơn khi `r` tăng.

### 4.3. Công thức chuẩn hóa công suất trong Chương 3 nên viết rõ hơn

Phương trình chuẩn hóa hiện là:

```tex
x_{\mathrm{norm}}=\frac{x_{\mathrm{raw}}}{\sqrt{\mathbb{E}\{|x_{\mathrm{raw}}|^2\}+\epsilon}}.
```

Công thức này chuẩn hóa công suất trung bình về xấp xỉ 1. Trước đó luận văn nêu ràng buộc:

```tex
\mathbb{E}\{|x|^2\}\le P_0.
```

Nếu muốn nhất quán với ràng buộc công suất `P_0`, nên viết một trong hai cách:

```tex
x_{\mathrm{norm}}
=
\sqrt{P_0}
\frac{x_{\mathrm{raw}}}
{\sqrt{\mathbb{E}\{|x_{\mathrm{raw}}|^2\}+\epsilon}}.
```

hoặc giải thích rằng trong các thí nghiệm AutoEncoder, `P_0=1`. Nếu không, người đọc sẽ thấy `P_0` xuất hiện trong ràng buộc nhưng biến mất trong công thức chuẩn hóa.

### 4.4. Danh mục ký hiệu còn thiếu nhiều ký hiệu dùng trong chương kỹ thuật

Luận văn có nhiều ký hiệu quan trọng nhưng danh mục ký hiệu/chữ viết tắt nên được bổ sung để giảm tải cho người đọc. Nên thêm hoặc kiểm tra các ký hiệu sau:

- `L_A`, `L_E`, `L_ch`, `L_APP`;
- `N_err`, `N_bit`, `N_sym,err`, `N_sym`;
- `I_max`;
- `E` là số cạnh trong Tanner graph;
- `d_in`, `d_h`, `d_out`;
- `N_theta`;
- `C_MLP`, `C_total`;
- `R_c`, `E_s`, `N_0`, `SNR`;
- `A_sat`, `p`, `alpha_PM`, `beta_PM`;
- `Delta f`, `T_s`;
- `CN(0, sigma^2)` nếu Chương 3 dùng nhiễu Gaussian phức.

Nếu danh mục ký hiệu quá ngắn, luận văn vẫn đọc được nhưng kém thân thiện và dễ bị phản biện nhận xét là chưa chuẩn hóa notation.

## 5. Vấn đề trình bày và dàn trang

### 5.1. Mục lục có dấu hiệu xuống dòng xấu ở tiêu đề Chương 3

Qua kiểm tra bản trích xuất và layout, tiêu đề:

```text
Chapter 3. DEEP LEARNING FOR INTELLIGENT MODULATION
```

có dấu hiệu bị bẻ thành nhiều dòng trong mục lục. Nếu PDF nhìn trực quan cũng bị bẻ dòng từng cụm từ ngắn, đây là lỗi trình bày đáng sửa vì mục lục là phần đầu tiên hội đồng nhìn thấy.

**Đề xuất sửa:** dùng tiêu đề ngắn cho mục lục:

```tex
\chapter[\textbf{DEEP LEARNING FOR INTELLIGENT MODULATION}]
{DEEP LEARNING FOR INTELLIGENT MODULATION}
```

nếu vẫn chưa đủ, có thể rút optional TOC title thành:

```tex
\chapter[\textbf{INTELLIGENT MODULATION}]
{DEEP LEARNING FOR INTELLIGENT MODULATION}
```

Hoặc điều chỉnh `tocloft` như tăng khoảng dành cho số chương, giảm bold trong TOC, hoặc cho phép dòng TOC rộng hơn.

### 5.2. LaTeX log còn nhiều `Underfull \hbox`

Log không cho thấy lỗi biên dịch nghiêm trọng, nhưng có rất nhiều cảnh báo `Underfull \hbox`. Đây thường không làm sai nội dung, nhưng là dấu hiệu một số đoạn, bảng hoặc dòng trong front matter bị căn dòng xấu, giãn chữ không đẹp, hoặc cột bảng quá hẹp.

**Đề xuất xử lý:** không cần sửa tất cả cảnh báo, nhưng nên kiểm tra trực quan các vùng sau:

- các trang front matter có dòng tên cơ quan/hội đồng;
- Bảng 2.1 và Bảng 2.2;
- các đoạn có nhiều thuật ngữ dài như `reliability-aware`, `ANN-assisted`, `check-node`;
- mục lục, danh mục hình, danh mục bảng.

Nếu bảng bị giãn chữ, nên dùng `tabularx` với cột rộng hơn, giảm bớt câu dài trong ô, hoặc tách bảng thành hai bảng nhỏ.

### 5.3. Bảng 2.1 và Bảng 2.2 hơi nhiều chữ trong từng ô

Hai bảng này có giá trị học thuật, nhưng đang chứa nhiều câu diễn giải dài. Trong luận văn kỹ thuật, bảng nên giúp người đọc quét nhanh. Nếu mỗi ô là một đoạn văn, bảng trở nên khó đọc.

**Đề xuất sửa:**

- Bảng 2.1 chỉ giữ cấu hình ngắn gọn: code, channel, decoders, iterations, model, training, SNR grid.
- Chuyển phần giải thích "trend-level evidence..." ra đoạn văn ngay dưới bảng.
- Bảng 2.2 chỉ giữ so sánh ngắn; phần "not included as plotted numerical baselines..." nên đưa vào chú thích dưới bảng hoặc đoạn văn sau bảng.

### 5.4. Nên kiểm tra độ phân giải và nhãn trục của các hình kết quả

Hình BER và training loss là bằng chứng chính của Chương 2. Cần đảm bảo:

- trục x/y có nhãn rõ và đơn vị rõ;
- legend đủ lớn khi in A4;
- đường BER phân biệt được khi in đen trắng;
- BER dùng thang log đúng;
- caption nêu đúng code, channel và decoder setting;
- nếu có vùng `10^{-5}`, hình cần đủ resolution để đọc được.

Nếu hình chỉ lấy từ screenshot hoặc export raster độ phân giải thấp, nên export lại từ Python/Matlab với DPI cao, ví dụ 300 hoặc 600 DPI.

### 5.5. Viết hoa thuật ngữ kỹ thuật cần thống nhất hơn

Luận văn đôi lúc viết `Modulation`, `Demodulation` giữa câu như tên khối hệ thống, đôi lúc lại dùng như danh từ thường. Đây không phải lỗi ngôn ngữ luận văn, nhưng là lỗi consistency văn phong khoa học.

**Đề xuất:** chọn một quy ước:

- dùng `modulation/demodulation` khi nói hiện tượng hoặc kỹ thuật chung;
- dùng `Modulation block`, `Demodulation block` khi nói khối chức năng trong sơ đồ hệ thống;
- dùng `AutoEncoder Modulation` nếu đó là tên hướng nghiên cứu/cụm thuật ngữ riêng.

## 6. Vấn đề flow trình bày

### 6.1. Chương 2 đang có phần kết quả trước khi các giới hạn khoa học được giải thích đầy đủ

Flow hiện tại của Chương 2 là: nền tảng LDPC -> MLP -> training/labeling -> performance -> complexity -> thảo luận mở rộng. Cách này đọc được. Tuy nhiên, vì kết quả enhanced-label có khả năng gây tranh luận, nên phần định nghĩa nhãn và giới hạn oracle nên được đặt gần hơn với đoạn kết quả.

**Đề xuất:** ngay trước hoặc ngay sau Hình 2.3, nên có một đoạn rất rõ:

> The enhanced-label curve should be interpreted as a reliability-control experiment. It is not claimed as a universally superior replacement for SPA unless the label-generation rule and test protocol are reproduced under the same conditions.

Câu này giúp người đọc không hiểu quá mức kết quả.

### 6.2. Chương 3 nên được định vị rõ là supporting study

Chương 3 hiện thiên về AutoEncoder modulation và non-ideal channels. Nội dung này liên quan đến nguồn LLR cho BIBCM-ID, nhưng không trực tiếp chứng minh ANN-LDPC decoder. Luận văn đã có câu "supporting evidence", nhưng nên nhắc lại ở đầu và cuối Chương 3 để hội đồng không hỏi vì sao Chương 3 không nối trực tiếp vào Hình 2.3.

**Đề xuất:** mở đầu Chương 3 nên viết rõ:

> This chapter is not presented as a second independent main contribution of the thesis. It supports the receiver-level motivation by showing how learned modulation/demodulation can affect the reliability of soft information supplied to the LDPC decoder.

Nếu muốn Chương 3 mạnh hơn, cần bổ sung thí nghiệm nối đầu ra demapper/AutoEncoder vào LDPC decoder. Nếu chưa có thí nghiệm đó, cách tốt nhất là giới hạn phạm vi.

### 6.3. Kết luận cuối luận văn nên tách rõ "đóng góp đã chứng minh" và "hướng mở rộng"

Phần kết luận hiện đã nêu đóng góp chính, nhưng nên tách rõ:

- đã chứng minh bằng mô phỏng: ANN check-node approximation trên QC-LDPC `(332,664)`, BPSK/AWGN;
- đã phân tích/hỗ trợ về mặt hệ thống: learned modulation/demodulation dưới PA/CFO/fading;
- chưa chứng minh đầy đủ: tích hợp end-to-end BIBCM-ID với learned demapper và ANN-LDPC decoder.

Cách tách này giúp luận văn trung thực hơn và tránh bị đánh giá là claim vượt dữ liệu.

## 7. Vấn đề văn phong khoa học

### 7.1. Nên giảm các cụm khẳng định mạnh khi chưa có thống kê đầy đủ

Các cụm như "improves clearly", "provides evidence", "can outperform SPA" nên luôn kèm điều kiện. Cách viết tốt hơn:

- Không nên: "The enhanced ANN can outperform SPA."
- Nên: "Under the evaluated QC-LDPC, BPSK/AWGN, and finite-simulation setting, the enhanced-label ANN shows a favorable BER trend relative to SPA."

Việc dùng câu có điều kiện không làm luận văn yếu đi. Ngược lại, nó thể hiện tác giả hiểu giới hạn của mô phỏng.

### 7.2. Nên tránh lặp lại cùng một ý quá nhiều lần

Một số ý được nhắc nhiều lần:

- ANN thay thế tanh/inverse tanh;
- enhanced label giống damping;
- result là trend-level;
- BPSK/AWGN là controlled setting.

Các ý này đúng, nhưng nếu lặp ở nhiều đoạn gần nhau thì văn phong có cảm giác phòng thủ. Nên giữ mỗi ý ở vị trí chiến lược:

- motivation ở đầu mục;
- evidence ở phần kết quả;
- limitation ở cuối mục hoặc discussion.

### 7.3. Nên dùng câu ngắn hơn ở các đoạn mô tả kỹ thuật dài

Một số câu trong Chương 2 và Chương 3 có nhiều mệnh đề liên tiếp. Khi viết tiếng Anh khoa học, câu dài không sai, nhưng với luận văn kỹ thuật có nhiều ký hiệu, câu dài làm người đọc khó theo dõi.

**Đề xuất:** chia các câu mô tả kết quả thành 2--3 câu ngắn:

- câu 1: nêu kết quả;
- câu 2: nêu điều kiện mô phỏng;
- câu 3: nêu giới hạn diễn giải.

Ví dụ:

> The enhanced-label ANN shows an approximate 0.2--0.4 dB horizontal shift relative to SPA near the low-BER waterfall region. This observation is obtained for the QC-LDPC `(332,664)` code under BPSK/AWGN and the stated training setup. Because the current figure does not include per-SNR error counts, the result is interpreted as trend-level evidence.

## 8. Đánh giá từng chương

### Chương 1

Chương 1 có vai trò đặt nền tảng tốt hơn trước. Các khái niệm BIBCM-ID, LLR, extrinsic information và iterative processing đã có mạch. Điểm cần bổ sung chủ yếu là định nghĩa thuật ngữ BIBCM-ID so với BICM-ID, quy ước LLR/BPSK, và giới hạn phạm vi interleaver.

**Mức độ cần sửa:** trung bình.  
**Rủi ro nếu không sửa:** phản biện có thể hỏi về sự lệch giữa bối cảnh BIBCM-ID và thí nghiệm BPSK/AWGN.

### Chương 2

Đây là chương mạnh nhất và cũng là chương cần bảo vệ kỹ nhất. Nội dung ANN-LDPC đã có cấu trúc hợp lý: nền tảng LDPC, SPA/Min-Sum, MLP, training, BER, complexity, limitation. Tuy nhiên, các điểm cần chốt là `I_max`, error-count log, nguồn reference message, vector đầu vào bốn chiều và điều kiện diễn giải mức tăng 0.2--0.4 dB.

**Mức độ cần sửa:** cao.  
**Rủi ro nếu không sửa:** phản biện có thể đánh giá kết quả mô phỏng chưa đủ tái lập hoặc nhãn enhanced-label chưa đủ thực tế.

### Chương 3

Chương 3 có vai trò hỗ trợ cho modulation/demodulation side. Nội dung về AutoEncoder, PA, CFO, fading, ISI phù hợp với bối cảnh hệ thống, nhưng chưa nối định lượng trực tiếp với ANN-LDPC ở Chương 2. Vì vậy cần viết rõ đây là supporting study, không phải bằng chứng hoàn chỉnh cho toàn bộ BIBCM-ID receiver.

**Mức độ cần sửa:** trung bình.  
**Rủi ro nếu không sửa:** phản biện có thể xem Chương 3 là một nhánh rời, chưa tích hợp với đóng góp chính.

### Kết luận

Kết luận hiện đã đi đúng hướng khi nói ANN-LDPC là đóng góp chính và learned modulation là hỗ trợ. Nên tách rõ các claim đã được mô phỏng, các claim phân tích, và các hướng future work.

**Mức độ cần sửa:** thấp đến trung bình.  
**Rủi ro nếu không sửa:** kết luận có thể bị xem là hơi rộng hơn dữ liệu thực nghiệm.

## 9. Danh sách sửa nhanh theo mức ưu tiên

### Ưu tiên 1 - nên sửa trước khi gửi hội đồng

1. Thêm quy ước LLR và ánh xạ BPSK rõ ràng.
2. Ghi cụ thể `I_max` và trạng thái syndrome stopping.
3. Định nghĩa nguồn của "reference message" trong enhanced-label training.
4. Đặc tả vector đầu vào bốn chiều của MLP.
5. Bổ sung hoặc hạ mức claim BER nếu không có error-count log.

### Ưu tiên 2 - nên sửa để luận văn chuyên nghiệp hơn

1. Thêm bảng tái lập mô phỏng: frame/bit/error count theo SNR.
2. Chuẩn hóa `\label` và `\eqref` cho các phương trình chính.
3. Bổ sung danh mục ký hiệu.
4. Sửa mục lục nếu tiêu đề Chương 3 bị xuống dòng xấu.
5. Rút gọn nội dung dài trong Bảng 2.1 và Bảng 2.2.

### Ưu tiên 3 - cải thiện văn phong và flow

1. Giảm lặp các câu tự giới hạn.
2. Chia câu dài ở các đoạn kết quả.
3. Thống nhất viết hoa `modulation`, `demodulation`, `AutoEncoder Modulation`.
4. Định vị Chương 3 là supporting study ngay từ đầu chương.
5. Tách kết luận thành "validated results", "analytical support", "future work".

## 10. Kết luận phản biện

Bản luận văn hiện tại đã đủ nền để phát triển thành một bản nộp có chất lượng, đặc biệt ở Chương 2 về ANN-assisted LDPC decoding. Điểm mạnh là tác giả đã chuyển từ lối trình bày khẳng định chung sang cách diễn giải có điều kiện, có bảng cấu hình và có nhận thức về giới hạn mô phỏng. Đây là cải thiện quan trọng.

Điểm yếu còn lại là độ chặt tái lập và định nghĩa thuật toán. Một phản biện kỹ thuật sẽ không chỉ hỏi "kết quả có tốt không", mà sẽ hỏi "người khác có tái lập được không", "LLR có đúng dấu không", "nhãn enhanced-label có dùng thông tin không thực tế không", và "MLP bốn đầu vào áp dụng cho check-node degree nào". Nếu sửa được các điểm này, luận văn sẽ vững hơn nhiều khi bảo vệ.

Khuyến nghị cuối cùng: trước khi nộp bản tiếp theo, nên ưu tiên sửa Chương 2 trước, vì đây là nơi chứa đóng góp chính và cũng là nơi dễ bị hỏi nhất. Chương 1 và Chương 3 chủ yếu cần chỉnh để hỗ trợ framing và flow, còn Chương 2 cần chỉnh để tăng tính tái lập, tính chính xác và tính bảo vệ học thuật.
