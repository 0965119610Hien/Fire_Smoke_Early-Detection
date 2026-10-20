# Đánh giá fold 0 — Stage 1 + Stage 2 dense (2026-10-21)

Báo cáo đầy đủ (mọi bảng): [`eval_20261020-154030_dense10fps.md`](eval_20261020-154030_dense10fps.md) / `.json`.
Đây là lần train đầu tiên trên **bộ dữ liệu hợp nhất + chia MILP 5-fold** (xem `docs/Unified_Dataset_Statistics.md`),
với tăng cường dữ liệu như cũ, nên đóng vai trò **baseline** cho các thay đổi sau.

## 1. Thiết lập

| | |
|---|---|
| Dữ liệu | Train = fold 1–4, val = fold 0, test = phần test cố định (giống nhau cho mọi fold) |
| Stage 1 | `segformer_spatial_best_fold0.pth` — run `20261020-151616`, 40 epoch, chọn epoch 26 (val Fg-mIoU tĩnh 65,30%) |
| Stage 2 | `timesformer_dense_fold0_spatiotemporal_best.pth` — run `20261020-154030`, dense 10 FPS, 35 epoch, chọn epoch 6 (val Fg-mIoU video 46,43%), γ = +0,019 |
| Test video | 279 frame có nhãn / 17 video (12 dương tính, 3 âm tính khó, 2 âm tính thường) |
| Test ảnh tĩnh | 182 ảnh (Stage 1) |
| Ngưỡng báo động | (τ, A_min, k) chọn trên val fold 0: recall ≥ 95% với FAR thấp nhất, nếu không đạt thì F1 lớn nhất; cố định khi chấm test |
| CI 95% | bootstrap 1000 lần theo video (ảnh tĩnh: theo ảnh) |
| Code | git `7d36de1` (train: `baf8d05`) |

Ba cấu hình trên video: **Stage 1** chạy từng frame; **Stage 2 γ=0** (decode head đã fine-tune, tắt nhánh temporal);
**Stage 2 đầy đủ**.

## 2. Kết quả chính (test)

### Pixel (%)

| Cấu hình | IoU Lửa [CI] | IoU Khói [CI] | Fg-mIoU [CI] | All-mIoU |
|---|---|---|---|---|
| Stage 1 — ảnh tĩnh (182 ảnh) | 69,51 [59,3; 76,0] | 56,73 [48,3; 64,6] | **63,12** [56,0; 68,9] | 74,19 |
| Stage 1 — frame video | 57,47 [49,6; 60,4] | 24,20 [7,2; 52,9] | 40,83 [31,5; 54,4] | 58,66 |
| Stage 2 γ=0 | 62,78 [48,4; 69,2] | 25,30 [8,1; 52,8] | 44,04 [33,0; 58,1] | 60,79 |
| **Stage 2 đầy đủ** | **62,87** [46,2; 70,9] | **30,14** [13,7; 51,6] | **46,51** [33,9; 56,7] | **62,01** |

### Báo động theo frame (ngưỡng từ val)

| Cấu hình | Lửa Recall / FAR | Khói Recall / FAR | Bất kỳ: Recall / FAR | MCC (lửa / khói / bất kỳ) |
|---|---|---|---|---|
| Stage 1 — ảnh tĩnh | 96,4 / 4,2 | 88,2 / 12,3 | 95,0 / 34,8 | 0,920 / 0,754 / 0,602 |
| Stage 1 — frame video | 91,3 / 3,1 | 65,7 / 26,1 | 91,0 / 31,3 | 0,880 / 0,386 / 0,614 |
| Stage 2 γ=0 | 91,3 / **0,8** | 67,7 / **21,1** | 88,9 / 32,5 | **0,903** / 0,460 / 0,573 |
| Stage 2 đầy đủ | 87,3 / 1,6 | **73,7** / 24,4 | 87,4 / 36,3 | 0,856 / **0,478** / 0,520 |

### Theo video và theo vùng (object)

| Cấu hình | Video lửa: Recall / FAR | Video khói: Recall / FAR | Vùng lửa LTPR / LPPV | Vùng khói LTPR / LPPV | Số vùng khói dự đoán |
|---|---|---|---|---|---|
| Stage 1 | 10/11 / 0/6 | 8/9 / **4/8** | 64,5 / 18,6 | 29,3 / 5,7 | 684 |
| Stage 2 γ=0 | 10/11 / 0/6 | 7/9 / 4/8 | 70,2 / 47,2 | 30,1 / 6,0 | 667 |
| Stage 2 đầy đủ | 10/11 / 0/6 | 7/9 / 4/8 | 65,7 / **51,8** | **41,4 / 28,4** | **190** (GT: 133) |

### Tốc độ (GPU A10, fp16)

Tiền xử lý CPU 14,2 ms/frame (frame ≤ 960 px; trước đây 74 ms ở 1080p) · Stage 1 streaming 62,6 FPS ·
Stage 2 streaming có cache 35,7 FPS → đủ thời gian thực cho 1 luồng 25–30 FPS.

## 3. Quá trình train

| | Stage 1 (ảnh tĩnh) | Stage 2 dense (video) |
|---|---|---|
| Val Fg-mIoU tốt nhất | 65,30% (epoch 26) | 46,43% (epoch 6) |
| Val Fg-mIoU từ đó đến cuối | đi ngang 64–65% từ epoch 15 | giảm dần còn ~41,5% |
| Val loss | thấp nhất 0,745 (ep 16) → 1,32 (ep 40) | thấp nhất 1,155 (ep 4) → 2,14 (ep 35) |
| Train loss | 1,11 → 0,16 | 1,34 → 0,29 |
| γ (nhánh temporal) | – | dương, tăng đều 0,010 → 0,029 |

## 4. Nhận xét

1. **Lửa đã tốt, khói là điểm yếu chính.** Báo động lửa theo frame có recall 87–91% với FAR chỉ 0,8–3% (MCC ~0,9); theo video
   phát hiện 10/11 video có lửa, không báo nhầm video nào. Khói thì IoU chỉ 24–30% trên video, recall frame 66–74% với FAR 21–26%,
   và **4/8 video không có khói bị báo khói**.

2. **Nhánh temporal giờ có ích** — khác hẳn lần đánh giá 19/10 (khi đó γ âm và bật temporal luôn làm kết quả xấu đi).
   γ dương và tăng dần trong lúc train; Stage 2 đầy đủ tốt nhất về pixel (Fg-mIoU 46,5 so với 44,0 và 40,8) và đặc biệt về **vùng khói**:
   số vùng khói dự đoán giảm từ 667 xuống 190 (GT 133), độ chính xác vùng (LPPV) tăng từ 6% lên 28%, phát hiện vùng (LTPR) từ 30% lên 41%.
   Nhánh temporal giúp khói "liền mạch" hơn, bớt các mảng lốm đốm. Khả năng cao nhờ dữ liệu đã được thống nhất (không còn CLAHE hai lần /
   kéo giãn ở round 2) và frame ngữ cảnh lấy từ video gốc, nhưng chưa tách riêng được nguyên nhân.
   *Đổi lại*, báo động "bất kỳ" của Stage 2 đầy đủ có FAR cao hơn (36% so với 31–33%); mọi chênh lệch đều nằm trong khoảng tin cậy rất rộng.

3. **Fine-tune decode head trên video (γ=0) cải thiện so với Stage 1**: IoU lửa 57,5 → 62,8, FAR lửa 3,1% → 0,8%, LPPV lửa 18,6% → 47,2%
   (bớt vùng lửa giả). Với khói thì gần như không đổi.

4. **Khoảng cách miền ảnh tĩnh → video rất lớn.** Cùng một Stage 1: Fg-mIoU 63,1% trên ảnh tĩnh nhưng 40,8% trên frame video; riêng khói
   56,7% → 24,2%. Stage 1 chỉ được train trên ảnh tĩnh (khói thường to, rõ: trung vị 6–7% diện tích), trong khi khói trong video test
   thường mỏng, xa, chất lượng thấp. Đây là lý do lớn nhất khiến khói kém.

5. **Âm tính khó là nguồn báo khói giả chính.** `neg_hard_negatives_jKMNSvpB9dY` (hơi nước bốc từ nồi ngoài trời lạnh) bị báo khói ở
   15/16 frame, `zFDB6XlHWVU` (sương máy phun ẩm) 7/16, `nJe-Y9QM0cA` 5/16. Mô hình chưa phân biệt được hơi nước / sương với khói.
   Ngoài ra `13106082` (đèn khò, chỉ có lửa) bị báo khói ở 8/8 frame — lỗi "đoán khói quanh lửa" đã thấy từ trước vẫn còn.

6. **Các video dương tính khó:** `electrical_ignition_VimYV-zgYo0` (khói mỏng ở xa, ảnh 650×480 mờ, có chữ chèn): IoU lửa 0, bỏ sót 7 frame
   lửa và 12 frame khói; `13862697` (lửa nhỏ, 4K thu nhỏ): IoU lửa 9%, bỏ sót 5 frame. Đều là ca vật thể nhỏ / mờ.

7. **Overfit mạnh, đặc biệt Stage 2.** Stage 2 đạt tốt nhất ở epoch 6 rồi val loss gần gấp đôi; 29 epoch sau là lãng phí GPU. Stage 1 đi ngang
   từ epoch 15. Tăng cường dữ liệu hiện chỉ có lật ngang + độ sáng (+ blur ở Stage 1), quá nhẹ so với ~1.300 frame / ~850 ảnh train.

8. **Độ tin cậy của số liệu.** Test chỉ có 17 video → CI rất rộng (Fg-mIoU Stage 2 đầy đủ: 33,9–56,7%), không chênh lệch nào giữa các cấu hình
   có ý nghĩa thống kê. Đây mới là **1 fold**; cần đủ 5 fold để có trung bình ± độ lệch chuẩn khi chọn cấu hình.

9. **Không so trực tiếp với số liệu 19/10** (Fg-mIoU 62,1%): tập test, tập val và cả quy trình tiền xử lý đều đã đổi; tập val Stage 1 cũ
   còn rò rỉ ảnh gần trùng với train. Số liệu hôm nay là mốc mới.

## 5. Đề xuất (theo ưu tiên)

1. **Dừng sớm + chọn checkpoint ổn định:** patience ~5 epoch theo val Fg-mIoU (hoặc val loss), giảm số epoch Stage 2 xuống ~15.
   Tiết kiệm GPU khi chạy 5 fold.
2. **Tăng cường dữ liệu mạnh hơn** (đã đề xuất): scale-crop 0,5–1,5×, nén JPEG / nhiễu / mờ chuyển động / hạ độ phân giải kiểu CCTV,
   gamma / bão hoà; Stage 2 áp dụng nhất quán cho cả clip.
3. **Thu hẹp khoảng cách miền cho Stage 1:** train Stage 1 trên ảnh tĩnh **cộng** frame có nhãn của các video train (cùng fold), để
   backbone thấy khói mỏng / xa ngay từ Stage 1 (Stage 2 đang đóng băng backbone).
4. **Bổ sung âm tính khó dạng hơi nước / sương / mây / bụi** (video + ảnh tĩnh, mask toàn 0) và cân nhắc tăng trọng số FP khói
   (class weight hoặc focal loss) — nhắm thẳng vào nhận xét 5.
5. Sau khi chốt 1–4 trên fold 0: chạy đủ **5 fold** và báo cáo trung bình ± độ lệch chuẩn; chỉ đánh giá test một lần cho cấu hình cuối.
