# LuanVan_Organized

Thu muc nay la ban sao da sap xep de phuc vu viet va nop luan van. Du lieu goc trong thu muc cha khong bi di chuyen hay doi ten.

## Cach dung nhanh

- `00_Admin_Template`: mau bieu, quy dinh dinh dang, quyet dinh hoi dong.
- `01_Manuscript`: ban thao luan van theo LaTeX, Word, PDF va file build.
- `02_Chuyen_De`: tai lieu chuyen de, hinh anh chuyen de, file render kiem tra.
- `03_Publications`: minh chung va source bai bao khoa hoc.
- `04_Code_Simulation`: code MATLAB/Python, Autoencoder modulation, notebook mo phong.
- `05_Results`: ket qua da gom rieng theo loai: BER, config, hinh, log, model weights.
- `06_Thesis_Assets`: hinh va bang nen uu tien dua vao luan van.
- `99_Archive_Do_Not_Edit`: noi du phong khi can cat giu cac ban cu.

## Ban thao trung tam

Nen dung `01_Manuscript/LaTeX/LuanVan_De.tex` lam ban goc neu tiep tuc viet theo LaTeX. Ban Word/PDF trong `01_Manuscript` nen xem la ban xuat de nop/doi chieu.

## Ket qua thuc nghiem

Trong `05_Results`, cac file co tien to theo thu muc thi nghiem de tranh ghi de, vi nhieu bo ket qua deu co ten goc `ber_summary.csv` va `config.json`.

## Goi nop luan van de xuat

Khi nop ban cuoi, uu tien lay:

- File Word/PDF tu `01_Manuscript`.
- Hinh chon loc tu `06_Thesis_Assets/Figures_Selected`.
- Bang ket qua tu `06_Thesis_Assets/Tables_Selected`.
- Bai bao khoa hoc tu `03_Publications/Publication_2/Manuscript`.
- Mau bieu hanh chinh tu `00_Admin_Template`.

Khong can nop `.venv`, `__pycache__`, log chi tiet, file build LaTeX, tru khi hoi dong yeu cau goi tai lap ket qua.

