# Sap xep noi dung phuc vu viet luan van cao hoc

Ngay lap: 2026-05-12

## 1. Nhan dinh tong quan

Thu muc hien co chua 5 nhom noi dung chinh:

1. **Ban thao luan van**: cac file `LuanVan_De.*`, trong do `LuanVan_De.tex` la nguon chinh, `LuanVan_De.pdf` va `LuanVan_De.docx` la ban xuat.
2. **Chuyen de nghien cuu**: thu muc `Chuyen de nghien cuu` gom ban chuyen de DOCX, file tao chuyen de, hinh anh, va anh render tung trang.
3. **Bai bao khoa hoc**: `publication 2` gom manuscript PDF, source LaTeX, hinh anh va anh tac gia; `publication 1` hien chi co anh chup man hinh.
4. **Ma mo phong va ket qua**: `BIBCM-ID-20260430T042613Z-3-001` gom code MATLAB cu, code Python, notebook, log, file mo hinh `.h5`, bang ket qua `.csv`, du lieu duong cong `.npz`.
5. **Mau bieu/quy dinh**: `MAU BIEU LUAN VAN, DE AN THAC SI`, cac file `M2 2026...`, `Mau LVTS.docx`, `rules.txt`, `template_rules.*`.

Nen coi `LuanVan_De.tex` la ban thao trung tam. Cac thu muc con la nguon dan chung, hinh, ket qua, va phu luc.

## 2. Phan loai theo muc dich viet bao cao

### 2.1. Ho so hanh chinh va dinh dang

Nguon hien co:

- `Mau LVTS.docx`
- `M2 2026_Mau Luan van thac si.docx`
- `MAU BIEU LUAN VAN, DE AN THAC SI/*`
- `2026.05 BMTT Quyet dinh Hoi dong chuyen de nghien cuu K36_Final.doc`
- `rules.txt`, `template_rules.txt`, `template_rules.json`

Cach dung:

- Dung de kiem tra bo cuc, trang bia, loi cam doan, loi cam on, muc luc, danh muc hinh/bang, phu luc, mau xac nhan chinh sua.
- Khong nen tron voi noi dung khoa hoc. Dat trong nhom `00_Admin_Template`.

### 2.2. Ban thao luan van

Nguon hien co:

- `LuanVan_De.tex`: nguon noi dung chinh.
- `LuanVan_De.pdf`: ban xem/bao ve noi dung.
- `LuanVan_De.docx`, `LuanVan_De_Formatted.docx`: ban Word xuat tu LaTeX/Pandoc.
- `LuanVan_De.aux`, `.log`, `.toc`, `.lof`, `.lot`, `.out`: file phat sinh khi bien dich.

Cach dung:

- Chi sua noi dung khoa hoc trong `LuanVan_De.tex` neu van tiep tuc theo LaTeX.
- Neu nop Word, dung `LuanVan_De_Formatted.docx` lam ban nop, nhung van nen giu `LuanVan_De.tex` lam ban goc de tranh mat cau truc.
- File phat sinh co the dua vao `build/` hoac bo qua khi sao luu.

### 2.3. Chuyen de nghien cuu

Nguon hien co:

- `Chuyen de - Ung dung Deep Learning cho anh xa dieu che trong he thong BIBCM-ID.docx`
- `1 Nghien cuu ky thuat hoc sau cho giai ma sua sai.docx`
- `Chuyen de nghien cuu/images/*`
- `Chuyen de nghien cuu/rendered_chuyen_de/page-*.png`
- `Chuyen de nghien cuu/build_chuyen_de.py`

Cach dung:

- Noi dung chuyen de phu hop de dua vao chuong tong quan va chuong de xuat mo hinh Autoencoder.
- Thu muc `images` la nguon hinh tot cho cac muc: so do he thong, Autoencoder, CFO, PA, constellation, BER theo cac kich ban.
- `rendered_chuyen_de` chi la anh kiem tra layout, khong phai nguon noi dung chinh.

### 2.4. Bai bao khoa hoc

Nguon hien co:

- `publication 2/REV_Journal_Final_Manuscript.pdf`
- `publication 2/latex_source/REV_Journal_Final.tex`
- `publication 2/latex_source/images/*`
- `publication 2/latex_source/images/authors/*`
- `publication 1/Screenshot *.png`

Cach dung:

- Bai bao `publication 2` trung voi phan Autoencoder-based modulation, co the dua vao muc "Bai bao khoa hoc" va trich noi dung cho chuong 3.
- Cac hinh trong `publication 2/latex_source/images` gan nhu trung voi `Chuyen de nghien cuu/images`; nen chon mot thu muc lam nguon chinh de tranh lech phien ban. De xuat dung `publication 2/latex_source/images` neu bai bao la ban moi hon.
- `publication 1` can bo sung ten bai bao/tep manuscript neu co; hien tai chi du de lam minh chung qua anh chup.

### 2.5. Ma nguon, mo phong va ket qua

Nguon hien co:

- `BIBCM-ID-20260430T042613Z-3-001/BIBCM-ID`: code MATLAB BIBCM-ID goc, interleaver, encoder, decoder, demapper, file cau hinh `.mat`.
- `BIBCM-ID-20260430T042613Z-3-001/python_code`: code Python mo phong BIBCM-ID, kenh, dieu che, giai dieu che, ma hoa/giai ma, worker, utils.
- `BIBCM-ID-20260430T042613Z-3-001/AE_Modulation`: code va notebook Autoencoder modulation, CFO multi-symbol, adaptive rate.
- `*_results/ber_summary.csv`: bang ket qua BER nen uu tien trich vao chuong thuc nghiem.
- `*_results/config.json`: cau hinh thuc nghiem, can ghi lai trong bang thong so mo phong.
- `*.weights.h5`: trong so mo hinh neural demapper, dung lam bang chung tai lap ket qua.
- `logs/*.log`, `logs/*.jsonl`: nhat ky chay, dung de truy vet neu can.

Cach dung:

- Chuong phuong phap: tham chieu `bicm_id.py`, `encoders.py`, `decoders.py`, `demodulation.py`, `channel.py`, `AE_Modulation/*.py`.
- Chuong thuc nghiem: uu tien `ber_summary.csv`, `config.json`, cac hinh `ber_*.png` va hinh trong `images`.
- Phu luc: dua danh sach notebook, cach chay mo phong, va cau truc code.
- Khong dua `.venv` va `__pycache__` vao phu luc hay sao luu hoc thuat.

## 3. Cau truc chuong luan van de xuat

1. **Mo dau**
   - Ly do chon de tai: BIBCM-ID, truyen thong so trong kenh khong ly tuong, nhu cau ung dung deep learning.
   - Muc tieu: cai tien/phan tich BIBCM-ID, neural demapper/check-node update, Autoencoder modulation, danh gia BER.
   - Doi tuong va pham vi: kenh Rayleigh, PA nonlinear, CFO, neural FEC/decoder.

2. **Co so ly thuyet ve BIBCM-ID va kenh truyen**
   - Lay tu chuong `BIT-INTERLEAVED CODED MODULATION WITH ITERATIVE DECODING`.
   - Bo sung LDPC/RSC/interleaver, 16-QAM, LLR, iterative decoding.

3. **Ung dung hoc sau trong dieu che va giai ma**
   - Lay tu chuyen de va `publication 2`.
   - Trinh bay Autoencoder, neural demapper, CFO compensation, PA distortion, Rayleigh/ISI.

4. **Mo hinh de xuat va thiet lap mo phong**
   - Mo ta pipeline BIBCM-ID + neural components.
   - Trich cau hinh tu `config.json` va code Python.
   - Dua bang thong so: modulation order, SNR/EbN0 range, channel model, so lan lap, loss, optimizer, batch size.

5. **Ket qua va danh gia**
   - Uu tien cac bang `ber_summary.csv` va hinh BER.
   - So sanh: conventional BIBCM-ID, neural residual demapper, robust PA, sequence CPE, sequence memory PA, Autoencoder vs 16-QAM.
   - Danh gia theo BER, do phuc tap, kha nang chiu CFO/PA, kha nang tai lap.

6. **Ket luan va huong phat trien**
   - Tom tat dong gop.
   - Han che: kich thuoc thuc nghiem, gia dinh kenh, tinh tong quat voi phan cung thuc.
   - Huong tiep: mo hinh kenh thuc, SDR/FPGA, ma hoa dai hon, hoc thich nghi online.

7. **Phu luc**
   - Danh sach code/notebook.
   - Bang cau hinh mo phong.
   - Bai bao khoa hoc.
   - Mau bieu/xac nhan neu truong yeu cau.

## 4. Cau truc thu muc de xuat khi don dep

Neu can sap xep lai, nen tao ban sao theo cau truc sau thay vi di chuyen truc tiep file goc:

```text
LuanVan_Organized/
  00_Admin_Template/
    Mau_bieu/
    Quyet_dinh/
    Quy_dinh_dinh_dang/
  01_Manuscript/
    LaTeX/
    Word/
    PDF/
    Build_Files/
  02_Chuyen_De/
    Docx/
    Source_Scripts/
    Rendered_Pages/
  03_Publications/
    Publication_1/
    Publication_2/
      Manuscript/
      Latex_Source/
      Images/
  04_Code_Simulation/
    MATLAB_BIBCM_ID/
    Python_BIBCM_ID/
    AE_Modulation/
    Notebooks/
  05_Results/
    BER_Tables/
    Figures/
    Model_Weights/
    Logs/
    Configs/
  06_Thesis_Assets/
    Figures_Selected/
    Tables_Selected/
    References/
  99_Archive_Do_Not_Edit/
```

## 5. Viec nen lam tiep theo

1. Chon **mot ban goc duy nhat** cho luan van: de xuat `LuanVan_De.tex`.
2. Chon **mot thu muc hinh chinh**: de xuat `publication 2/latex_source/images` neu bai bao la ban cap nhat nhat.
3. Trich cac file `ber_summary.csv` thanh bang tong hop cho chuong ket qua.
4. Tao bang mapping "hinh nao vao muc nao" de tranh lap hinh.
5. Don cac file phat sinh LaTeX (`.aux`, `.log`, `.toc`, `.lof`, `.lot`, `.out`) vao nhom build hoac bo qua khi nop.
6. Khong dua `.venv`, `__pycache__`, file log qua chi tiet vao goi nop luan van; chi giu trong goi tai lap ket qua neu can.

## 6. Bang gan noi dung vao chuong

| Noi dung | Nguon nen dung | Dua vao chuong |
|---|---|---|
| Tong quan BIBCM-ID, LDPC, iterative decoding | `LuanVan_De.tex` chuong BIBCM-ID | Chuong 2 |
| Neural update/check node, neural demapper | `python_code`, `LuanVan_De.tex` chuong ANN | Chuong 3-4 |
| PA nonlinear, CFO, Rayleigh, ISI | `publication 2/latex_source/REV_Journal_Final.tex`, hinh `images` | Chuong 3 |
| Autoencoder modulation | `AE_Modulation`, `publication 2`, `Chuyen de nghien cuu` | Chuong 3-4 |
| Ket qua BER BIBCM-ID | `python_code/ber_*.png`, `logs`, `ber_summary.csv` | Chuong 5 |
| Ket qua AE vs 16QAM | `Scenario *.png`, notebook AE | Chuong 5 |
| Bang thong so mo phong | `*_results/config.json`, `config.py` | Chuong 4 |
| Bai bao khoa hoc | `publication 2/REV_Journal_Final_Manuscript.pdf` | Phu luc/Bai bao |
| Mau bieu nop | `MAU BIEU...`, `M2 2026...` | Ho so nop |

