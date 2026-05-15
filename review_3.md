# Review 3 - Nhận xét phản biện sau lần rà soát mới nhất

File được rà soát: `latex/thesis.pdf`  
Bản PDF hiện tại: 93 trang, tạo lúc 15/05/2026 10:07:09.

## Nhận xét tổng quan

Bản luận văn hiện tại đã cải thiện đáng kể so với bản 89 trang trước đó. Các lỗi lớn đã được xử lý một phần:

- Caption mang dấu vết biên tập như `redrawn in English` đã được sửa.
- Chương 2 đã có hình và bảng riêng: sơ đồ MLP, training loss, BER comparison, bảng cấu hình mô phỏng/training, bảng tiêu chí so sánh decoder.
- Công thức 49 tham số của MLP đã được giải thích rõ hơn với `d_in = 4`, `d_h = 8`, `d_out = 1`.
- Chương 3 đã đổi các tiêu đề mang tính tự biện hộ thành các tiêu đề học thuật hơn như `Relevance to LDPC Decoder Input Reliability`, `Validation Path for Soft-Information Integration`, `Integrated Interpretation for BIBCM-ID-Oriented Receivers`.
- Kết luận đã bổ sung một số thông tin định lượng hơn: QC-LDPC `(332,664)`, BPSK/AWGN, 49 tham số, gain khoảng `0.2-0.4 dB`.

Về tổng thể, flow trình bày hiện đã tốt hơn. Chương 1 đặt nền tảng hệ BIBCM-ID và chất lượng thông tin mềm. Chương 2 tập trung vào đóng góp chính: ANN hỗ trợ cập nhật check-node trong LDPC decoding. Chương 3 đóng vai trò bổ trợ, phân tích điều chế/giải điều chế học sâu như nguồn tạo LLR cho decoder. Cách nối này hợp lý và đủ bảo vệ về mặt cấu trúc.

Tuy nhiên, bản hiện tại vẫn còn một số lỗi kỹ thuật và điểm yếu học thuật cần sửa trước khi coi là bản hoàn chỉnh.

## 1. Rà soát văn phong khoa học

Văn phong hiện tại đã khá hơn. Các câu khẳng định được viết thận trọng hơn, tránh tuyên bố quá mức rằng toàn bộ hệ BIBCM-ID đã được tích hợp hoặc tối ưu hoàn chỉnh. Luận văn đã dùng các cụm như `reported conditions`, `bounded contribution`, `supporting analysis`, `not a universal replacement`, đây là cách viết phù hợp với mức độ bằng chứng hiện có.

Tuy nhiên vẫn còn một vài điểm nên chỉnh:

- Cụm `reported curve`, `reported simulations`, `reported decoder study` xuất hiện khá nhiều. Cụm này chấp nhận được, nhưng nếu lặp nhiều sẽ tạo cảm giác kết quả được “kể lại” thay vì được trình bày như kết quả chính thức của luận văn. Nên thay một phần bằng `the BER curve in Figure 2.3`, `the simulation in Table 2.1`, `the evaluated QC-LDPC scenario`.
- Câu `From an examination viewpoint...` ở mục 3.15 vẫn hơi mang giọng phản biện/nội bộ. Nên đổi thành giọng học thuật hơn, ví dụ: `From a system-level viewpoint, the three chapters form a single argument...`
- Một số đoạn ở Chương 2 sau khi đã có bảng/hình vẫn còn hơi thận trọng quá mức. Thận trọng là tốt, nhưng cần cân bằng với việc khẳng định rõ đóng góp của tác giả.

Đánh giá văn phong: hiện đạt mức khá, đủ nghiêm túc, nhưng nên sửa các cụm còn mang giọng “báo cáo phản biện” để bản luận văn tự nhiên hơn.

## 2. Rà soát flow trình bày

Flow lớn hiện đã ổn:

1. Mở đầu: đặt vấn đề truyền thông số, LDPC, BICM-ID/BIBCM-ID, ANN và AutoEncoder.
2. Chương 1: mô hình BIBCM-ID, LLR, extrinsic information, labeling, interleaving, vấn đề độ tin cậy thông tin mềm.
3. Chương 2: đóng góp chính về ANN-assisted LDPC decoding.
4. Chương 3: phân tích learned modulation/demodulation như nguồn ảnh hưởng đến LLR đầu vào LDPC.
5. Kết luận: quay lại ANN-LDPC là đóng góp chính, AutoEncoder là phần bổ trợ.

Điểm tốt nhất của bản mới là Chương 2 đã có bằng chứng trực quan và bảng cấu hình. Điều này làm flow bằng chứng tốt hơn nhiều so với bản trước.

Tuy nhiên, flow chi tiết còn hai điểm cần chú ý:

- Chương 2 hiện có phần cơ sở, kết quả, rồi sau đó lại có nhiều mục mở rộng như `Evaluation Protocol`, `Ablation View`, `Scientific Boundaries`. Các mục này tốt, nhưng cần đảm bảo không làm người đọc thấy phần kết quả chính bị “loãng”. Nên cân nhắc đưa một phần các mục này vào cuối chương như “Discussion” hoặc “Validation considerations”.
- Chương 3 vẫn hơi dài so với vai trò bổ trợ. Nếu cần rút gọn, nên giữ các phần trực tiếp liên quan đến LLR calibration, soft-output demodulation, PA/CFO ảnh hưởng LLR; giảm các đoạn giải thích cấu trúc luận văn.

Đánh giá flow: cấu trúc tổng thể đã ổn; cần tinh chỉnh để Chương 2 nổi bật hơn nữa là chương đóng góp chính.

## 3. Các điểm đã sửa tốt

### 3.1. Caption hình Chương 1

Đã sửa tốt. Danh mục hình hiện ghi:

- `Block diagram of the BIBCM-ID system model.`
- `Soft-information exchange in the iterative BIBCM-ID receiver.`

Không còn dấu vết `redrawn in English`.

### 3.2. Chương 2 đã có hình và bảng

Đã bổ sung:

- Hình 2.1: MLP compact cho check-node update.
- Bảng 2.1: cấu hình mô phỏng và training.
- Hình 2.2: training-loss curve.
- Hình 2.3: BER comparison.
- Bảng 2.2: tiêu chí so sánh decoder.

Đây là cải thiện rất quan trọng vì Chương 2 là đóng góp chính.

### 3.3. Công thức 49 tham số đã được làm rõ ở phần đầu Chương 2

Trong mục 2.2, luận văn đã nêu rõ cấu hình đại diện:

- `d_in = 4`
- `d_h = 8`
- `d_out = 1`
- Tổng tham số: `4 x 8 + 8 + 8 x 1 + 1 = 49`

Điểm này đã tốt hơn bản trước.

### 3.4. Kết luận đã có thêm định lượng

Kết luận hiện đã nêu:

- QC-LDPC `(332,664)`.
- Kênh BPSK/AWGN.
- MLP 49 tham số.
- ANN gần SPA, tốt hơn Min-Sum trong waterfall region.
- Enhanced-label gain khoảng `0.2-0.4 dB`.

Đây là hướng sửa đúng.

## 4. Các vấn đề còn cần sửa

### 4.1. Còn lỗi không nhất quán về `d_in = 4` và `d_in = 5`

Đầu Chương 2 đã sửa đúng là `d_in = 4` để ra 49 tham số. Tuy nhiên ở mục 2.10 `Complexity Accounting for ANN-Assisted LDPC Decoding`, PDF vẫn ghi:

`For d_in = 5, d_h = 8, and d_out = 1, this is about 48 multiplications...`

Vấn đề:

- Nếu `d_in = 5`, số MAC sẽ là `5 x 8 + 8 x 1 = 48`, nhưng số tham số sẽ là `5 x 8 + 8 + 8 x 1 + 1 = 57`, không phải 49.
- Nếu luận văn muốn giữ cấu hình 49 tham số, đoạn này phải dùng `d_in = 4`.
- Khi `d_in = 4`, số MAC xấp xỉ là `4 x 8 + 8 x 1 = 40`, không phải 48.

Đề nghị sửa:

`For d_in = 4, d_h = 8, and d_out = 1, this is about 40 multiply-accumulate operations per outgoing message, plus ReLU comparisons.`

Đây là lỗi quan trọng vì liên quan trực tiếp đến lập luận độ phức tạp.

### 4.2. Công thức số tham số ở mục 2.10 đang thiếu tích `d_h d_out`

Trong PDF, công thức ở mục 2.10 hiển thị:

`N_theta = d_in d_h + d_h + d_h + d_out`

Đúng ra phải là:

`N_theta = d_in d_h + d_h + d_h d_out + d_out`

Nếu công thức nguồn đã đúng nhưng PDF hiển thị sai, cần kiểm tra lại LaTeX. Nếu nguồn cũng sai, cần sửa ngay. Đây là lỗi kỹ thuật rõ ràng.

### 4.3. Bảng 2.1 vẫn còn thiếu số liệu đánh giá cụ thể

Bảng 2.1 đã là cải thiện lớn, nhưng dòng `Evaluation range` vẫn còn khá chung:

`SNR points around the waterfall region... interpreted together with number of simulated bits and observed errors`

Vấn đề: bảng chưa nêu số cụ thể về:

- SNR range chính xác.
- Số frame hoặc số bit mô phỏng.
- Số lỗi tối thiểu mỗi điểm BER.
- Số iteration tối đa.
- Có dùng stopping rule hay không.

Phản biện vẫn có thể hỏi: “Kết quả BER này dựa trên bao nhiêu bit? Gain 0.2-0.4 dB có đáng tin không?”

Đề nghị: nếu có dữ liệu, điền trực tiếp vào bảng. Nếu chưa có, nên thêm một câu thừa nhận giới hạn: `The current simulation is used for trend-level comparison; longer low-BER simulations are required for standard-level performance claims.`

### 4.4. Gain 0.2-0.4 dB vẫn chưa gắn với mức BER cụ thể

Bản mới đã nói gain nằm quanh `low-BER waterfall region`, nhưng vẫn chưa nêu mức BER cụ thể, ví dụ `at BER = 10^-3` hoặc `around BER = 10^-4`.

Đề nghị: nếu hình 2.3 cho phép đọc xấp xỉ, hãy viết rõ:

`The gain is approximately 0.2-0.4 dB around BER = ...`

Nếu không thể đọc chính xác, nên viết thận trọng hơn:

`The plotted curves indicate an approximate horizontal shift of 0.2-0.4 dB in the waterfall region; this should be treated as a trend-level gain unless longer simulations confirm the low-BER points.`

### 4.5. Bảng 2.2 nói NMS/OMS là reference nhưng chưa có đường NMS/OMS trong hình 2.3

Bảng 2.2 có dòng `NMS/OMS`, nhưng Hình 2.3 caption chỉ ghi so sánh ANN update, SPA và Min-Sum. Nếu không có đường NMS/OMS trong hình, cần làm rõ rằng NMS/OMS chỉ được thảo luận về mặt lý thuyết, chưa phải numerical baseline trong Hình 2.3.

Đề nghị một trong hai cách:

- Thêm đường NMS/OMS vào Hình 2.3 nếu có dữ liệu.
- Hoặc sửa bảng 2.2: `discussed as standard low-complexity references; not included in the plotted numerical comparison unless otherwise stated.`

### 4.6. Một số đoạn Chương 3 vẫn còn giọng “giải thích bố cục”

Tiêu đề đã đổi tốt, nhưng nội dung bên dưới vẫn còn câu:

`If Chapter 3 were written as a general deep-learning communication chapter, it would appear misaligned with the title.`

và:

`From an examination viewpoint, the three chapters should form one argument...`

Nhận xét: các câu này vẫn mang giọng tự biện hộ. Nên viết lại thành giọng khoa học hơn.

Đề nghị sửa:

Thay vì:

`If Chapter 3 were written as...`

Viết:

`The role of Chapter 3 is limited to the soft-information interface between learned demodulation and LDPC decoding.`

Thay vì:

`From an examination viewpoint...`

Viết:

`From a system-level viewpoint, the three chapters form a single argument...`

### 4.7. Một số phương trình trong PDF bị hiển thị chưa đẹp khi trích text, cần kiểm tra trực quan

Khi trích text, một số công thức có ký tự lạ hoặc bố trí khó đọc, ví dụ các công thức có dấu tổng, mũ, vector, quantizer. Đây có thể chỉ là lỗi `pdftotext`, nhưng nên kiểm tra trực quan trong PDF ở các trang:

- Trang 24-25: SPA, Min-Sum, NMS/OMS.
- Trang 37-42: box-plus, parameter count, quantization.
- Trang 50-55: LLR, Rapp model, CFO, Rayleigh multipath.

Nếu PDF nhìn đúng thì không cần sửa. Nếu PDF cũng vỡ, cần chỉnh LaTeX.

### 4.8. Tài liệu tham khảo vẫn nên rà metadata

Danh mục tham khảo đủ tốt hơn trước. Tuy nhiên vẫn nên kiểm tra:

- Tên tác giả có dấu nháy như `Be'ery` hiển thị nhất quán chưa.
- Bib key nội bộ `hu2001` nhưng bài năm 2005 không ảnh hưởng PDF, nhưng có thể gây nhầm nếu sau này bảo trì.
- Nếu Chương 3 nhấn mạnh calibration, nên cân nhắc thêm 1-2 nguồn về probability calibration/LLR calibration/neural demapper.

## 5. Đánh giá bản hiện tại

Bản hiện tại đã tiến gần hơn nhiều tới một bản luận văn có thể bảo vệ. So với bản review trước, các điểm trọng yếu đã được cải thiện:

- Chương 2 đã có bằng chứng hình/bảng.
- Flow Chương 3 hợp lý hơn.
- Kết luận có thông tin định lượng hơn.
- Caption và tiêu đề đã sạch hơn.

Điểm yếu chính còn lại là tính nhất quán kỹ thuật trong phần complexity và độ chắc của bằng chứng mô phỏng:

- Lỗi `d_in = 5` trong khi cấu hình chính là `d_in = 4`.
- Công thức số tham số ở mục 2.10 có khả năng sai.
- Gain 0.2-0.4 dB chưa gắn với mức BER/số bit/số lỗi cụ thể.
- NMS/OMS được nhắc trong bảng nhưng chưa rõ có nằm trong kết quả số hay không.

Đánh giá sơ bộ hiện tại: khoảng 7.2-7.6/10.  
Nếu sửa các lỗi nhất quán kỹ thuật ở Chương 2 và làm rõ số liệu mô phỏng, bản này có thể đạt khoảng 8.0/10.

## 6. Ưu tiên chỉnh sửa tiếp theo

1. Sửa toàn bộ chỗ `d_in = 5` thành `d_in = 4` nếu cấu hình chính là MLP 49 tham số.
2. Sửa công thức số tham số tại mục 2.10 thành `N_theta = d_in d_h + d_h + d_h d_out + d_out`.
3. Sửa số MAC tương ứng: với `d_in = 4`, `d_h = 8`, `d_out = 1`, số MAC xấp xỉ là 40.
4. Làm rõ gain 0.2-0.4 dB tại mức BER nào.
5. Bổ sung số frame/bit mô phỏng hoặc nói rõ đây là trend-level simulation.
6. Làm rõ NMS/OMS có phải numerical baseline trong Hình 2.3 không.
7. Viết lại các câu Chương 3 còn mang giọng “giải thích bố cục”.
8. Kiểm tra trực quan các công thức trong PDF.
9. Rà lại metadata tài liệu tham khảo.
