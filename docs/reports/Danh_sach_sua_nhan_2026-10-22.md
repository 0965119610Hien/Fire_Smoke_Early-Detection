# Danh sách nhãn cần sửa (2026-10-22)

Quy tắc áp dụng: **hơi nước / sương = nền**, **mọi ngọn lửa (kể cả lửa xanh bếp gas) = lửa**, **khói do cháy và khói xả có muội = khói**,
vùng mơ hồ = **255 (bỏ qua)**.

Cách tìm: `python scripts/find_label_issues.py` gắn cờ ứng viên theo heuristic (khói trắng, mask hình hộp, nhãn nhảy giữa
các frame liền nhau, toàn bộ video âm tính khó). Kết quả nằm ở `data/manual_check_unified/label_issue_candidates.csv`, ảnh xem nhanh ở
`data/manual_check_unified/label_issue_sheets/`. Sau đó xem bằng mắt từng ứng viên, cộng thêm các ca đã ghi trong báo cáo fold 0.

**Giới hạn:** heuristic không bắt được mọi lỗi. Ví dụ hơi nước trên chảo `14112316` không bị gắn cờ "khói trắng", và nhãn bị
thiếu một phần thì không heuristic nào phát hiện. Khoảng 1.800 frame / ảnh không bị gắn cờ thì chưa được xem.

Đường dẫn ảnh video: `data/dataset_video_unified/<video>/<frame>.jpg`; ảnh tĩnh: `data/Fire_Smoke.v4i.png-mask-semantic/train/<tên file>`.

## A. Sai chắc chắn — sửa ngay

### A1. Video dương tính

| Split | Video | Frame | Lỗi | Sửa |
|---|---|---|---|---|
| val | `14112316_1440_2560_60fps_pure` | cả 11 frame (000000–000600) | Hơi nước bốc từ chảo bị gán khói | Hơi nước → nền (hoặc 255 nếu lẫn khói thật). Giữ lửa, kể cả lửa xanh bếp gas |
| test | `316af409_pexels_heavy_smoke_18105251_pure` | 000030–000090, 000240–000360 (các frame có khói) | Hơi nước bốc từ nồi mì bị gán khói | Hơi nước → nền |
| val | `battery_thermal_runaway_ld00r-tEEJ4_continuous_trimmed` | 000550 chắc chắn; rà 000200–000525 | Phòng mù khói nhưng nhãn chỉ có lửa | Thêm khói |
| val | `battery_thermal_runaway_ld00r-tEEJ4_continuous_trimmed` | 000125, 000175 | Vùng chói trắng do nổ / phơi sáng bị gán lửa cả mảng lớn | Thu hẹp về phần lửa thật, phần chói → 255 |
| val | `electrical_ignition_qRoT41qla9I_continuous_trimmed` | 000450 | Lửa nhìn rõ nhưng bị phủ hết bằng khói, không có nhãn lửa | Thêm lửa |
| train | `12218199_3114_1440_30fps_pure` | 000240 | Lửa rõ nhưng mask rỗng | Gán lửa |
| train | `battery_thermal_runaway_krHU3jlR4Pc` | 000425 | Lửa lớn chỉ được gán một dải mỏng dưới khối khói | Gán lại lửa |
| train | `36f3eac1_pexels_cigarette_smoke_5075501_pure` | 000125 | Khói vẽ thành khối chữ nhật dọc điếu thuốc, ảnh không có khói | Xoá / vẽ lại |
| train | `0aefc692_pexels_welding_sparks_4941457_pure` | 000075 | Khói trùm lên mặt nạ hàn và người thợ | Vẽ lại sát khói hàn hoặc 255 |
| train | `05ca4b9b_pexels_welding_sparks_34771762` | 000128 | Cột khói trùm cả người thợ hàn | Vẽ lại (xem thêm C4) |
| val | `9a06358f_pexels_heavy_smoke_3568453` | 000000, 000060, 000180 | Ống khói đang nhả khói nhưng nhãn rỗng | Vệt trắng từ ống khói → nền nếu là hơi nước, khói nếu xám; không chắc → 255 |

### A2. Video "âm tính khó" thực chất có khói thật (mâu thuẫn với nhãn rỗng)

| Split | Video | Nội dung | Sửa |
|---|---|---|---|
| train | `neg_hard_negatives_iaZ3E1mD96Q` | Lựu đạn khói, pháo sáng bắn tia lửa | Gán khói / lửa cho frame có khói, hoặc bỏ video khỏi tập âm tính |
| train | `neg_hard_negatives_BvTBuLESrJ0` | Thử khói nóng trong sân vận động (Fike) | Gán khói cho frame có cột khói, frame chữ giữ rỗng |
| train | `neg_hard_negatives_erqtaaYXWzY` | Khói xả xe xám / xanh / đen (đốt dầu) | Theo quy tắc khói xả có muội = khói → gán khói; frame chữ / hình vẽ giữ rỗng |

Các video âm tính khó còn lại (sương mù, hơi nồi, máy phun ẩm, máy phun sương diệt côn trùng, khí xả trắng trong tuyết
`r-9qPsm_pvc` / `nJe-Y9QM0cA`) **đúng với quy tắc**, giữ nguyên.

### A3. Ảnh tĩnh: hơi nước bị gán khói

| Split | File | Nội dung | Sửa |
|---|---|---|---|
| train | `000122_4-Image-No-4_jpg.rf.ff51100a53b2652245620e204a82f111.jpg` | Hơi xả từ van đường ống | Khói → nền |
| train | `000106_val_621_jpg.rf.9b1117f6f22ef485b46e2b379011aebc.jpg` | Nhà máy điện: vệt trắng từ tháp giải nhiệt | Vùng tháp giải nhiệt → nền; vệt từ ống khói xem C1 |
| train | `000170_train_3124_jpg.rf.35fa3cb3a450a5295277b12c5c90912b.jpg` | Như trên (cùng cảnh) | Như trên |
| train | `000135_train_3250_jpg.rf.28736bfbbf6a5759115a4409df65f553.jpg` | Nhìn từ trên: hơi bốc từ dàn quạt giải nhiệt | Khói → nền |
| val | `000254_train_1034_jpg.rf.7f2e0f4e28c43c3a8f054b86db58713c.jpg` | Như trên | Khói → nền |
| train | `000506_train_1080_jpg.rf.d89fcd6d8f6bd8d45e6dc6623142077e.jpg` | Như trên | Khói → nền |
| test | `000863_test_788_jpg.rf.32202857f9e141e73b4d75d0476fcb3e.jpg` | Như trên | Khói → nền |

## B. Lỗi nhẹ (nhãn thô)

| Nguồn | Mục | Lỗi |
|---|---|---|
| test | `12393269_1440_2560_30fps_pure` (cả video) | Lửa trong lò vẽ thành hình chữ nhật (từ báo cáo fold 0) |
| test | `electrical_ignition_VimYV-zgYo0_continuous_trimmed` frame cuối | Khói trùm dòng chữ "Smoke Detected!!" → chữ chèn = 255 |
| static train | `000464_2060_jpg.rf.d07e669d2606ddccabdfa3a952cf093e.jpg` | Khói vẽ theo từng ô kính, viền thô |
| train | `36f3eac1_..._pure` 000150, `28f4ec5d_..._pure` 000120 (test) | Mask khói thuốc lá trùm sang nền / cửa sổ |

## C. Cần bạn quyết định quy tắc trước (chưa thể gọi là sai)

1. **Khí thải công nghiệp từ ống khói** (thường là hơi nước, đôi khi có khói thật): static `000021_test_429_...`,
   `001085_train_1910_...`, `000453_train_3270_...`, vệt từ ống khói trong `000106` / `000170`; video `6c3dee00_pexels_heavy_smoke_6618036`
   (16 frame, đám mây trắng rất lớn bốc từ mái nhà xưởng nhìn từ drone); `9a06358f`. ~~Đề xuất: không rõ nguồn cháy → 255.~~
   **Đã chốt (2026-10-23): toàn bộ C1 → nền.** Đã làm trống mask (kèm `000106` / `000170`) qua
   `scripts/apply_label_overrides.py` + `data/label_overrides.csv`; mask gốc lưu ở `data/label_overrides_backup/`.
2. **Máy tạo khói / sương nhân tạo để thử đầu báo** (khói glycol, không phải do cháy):
   - "Super Fog": `000358_20test75_...`, `000755_20test75_...`
   - Kho "Material World": `000141_20test392_...` (val), `000507_20test413_...` (val), `000618_20test755_...`, `000945_20test740_...`,
     `001029_20test491_...`, `001229_20test488_...` (val)
   - "Altra-Air SMOKE TEST": `000940_25test167_...`, `001073_25test168_...`
   - Thử khói trong nhà kho / văn phòng: `000496_1test317_...`, `000739_16test345_...`, `001315_1test394_...`, `000509_2test781_...`,
     `001082_15test244_...`

   Về hình ảnh thì giống hệt khói và hữu ích cho việc học, nhưng theo quy tắc "sương = nền" thì mâu thuẫn. Đề xuất: **giữ là khói** và ghi
   ngoại lệ "khói nhân tạo dùng thử đầu báo = khói", vì mục tiêu là phát hiện thứ trông như khói đang lan trong nhà.

   **Đã chốt (2026-10-23): C2 → nền.** Tiêu chí là bản chất vật lý, không phải chủ ý: sương từ máy (glycol / dầu) là
   giọt ngưng tụ như hơi nước, nhất quán với C1. Viên / nến / que khói (đốt hoá chất, khói xám) vẫn là khói như lựu đạn
   khói. Cả 15 ảnh trên đều là máy tạo sương → đã làm trống mask qua `data/label_overrides.csv`.
3. **Người hút thuốc / vape trên camera an ninh**: `001303_ab169f92-...` (đám trắng lớn, nhiều khả năng là vape), `001257_d77040d3-...` (val),
   `001252_15c0eafe-...`, `000337_15c0eafe-...`. Vape là hơi glycol → theo quy tắc là nền; thuốc lá thật là khói. Đề xuất: vape → nền, khó phân biệt → 255.
4. **Hồ quang hàn** gán lửa trong `05ca4b9b_pexels_welding_sparks_34771762` (000128, 000160): hồ quang không phải ngọn lửa. Đề xuất:
   hồ quang + tia lửa hàn → nền (là âm tính khó tốt), khói hàn thật → khói.
   **Đã chốt (2026-10-23): C4 → nền** cho hồ quang, tia lửa và khói hàn. Đã làm trống toàn bộ frame có nhãn của
   `05ca4b9b` (22, gồm cả lửa gán trên hồ quang) và `0aefc692` (8) — action `empty_all`.
5. `000460_train_2914_...`: đám trắng trong đêm, không rõ là khói hay sương. **Đã chốt: khói, giữ nguyên nhãn.**

## Trạng thái (2026-10-23)

Đã áp toàn bộ qua `scripts/apply_label_overrides.py` (`data/label_overrides.csv`, 160 dòng; mask gốc ở `data/label_overrides_backup/`):

- A1, A2, B, C5: 84 mask lấy từ export Roboflow `data/fix_label.v1i.png-mask-semantic` (action `replace`).
- Ảnh không có trong export = nền: `316af409` 000360 + 5 ảnh A3 (action `empty`); 20 frame A2 vốn đã rỗng.
- C1, C2 → nền; C4 → nền (cả hồ quang gán lửa, cả 8 frame `0aefc692`); C3: `001303` (vape) → nền, 3 ảnh thuốc lá giữ khói.
- Quy tắc thuốc lá: mọi khói thuốc lá (đầu điếu và phả ra) = khói; vape = nền.

Sau khi áp — video 1864 frame: rỗng 538 / chỉ lửa 693 / cả hai 314 / chỉ khói 319; ảnh tĩnh 1058: rỗng 27 / chỉ lửa 550 /
cả hai 181 / chỉ khói 300. Ba video `neg_hard_*` của A2 giờ có khói / lửa nhưng `video_kind()` vẫn xếp theo tên.

## Thứ tự làm đề xuất

1. Chốt 5 điểm ở mục C.
2. Sửa A1 + A2 trước (video, ảnh hưởng trực tiếp tới số liệu khói và FAR), rồi A3.
3. Export lại từ Roboflow → `build_unified_dataset.py` → audit → train lại fold 0 để so với baseline.
