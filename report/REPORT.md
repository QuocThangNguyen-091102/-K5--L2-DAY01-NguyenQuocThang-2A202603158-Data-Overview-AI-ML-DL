# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** <điền>

**Runtime Colab:** <CPU hoặc GPU>

**Python / PyTorch / Ultralytics:** <điền từ output notebook>

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

1. Record hạng 1 (`sample_id = traffic`): `class_id`=<điền> ; `class_name`=<điền> ; `rank`=1 ; `score`=<điền> ; `taxonomy_name`=<điền>
2. Record này nói về **cả bức ảnh**. Phân loại ảnh gán một nhãn cho toàn bộ ảnh, không chỉ ra vị trí hay ranh giới của vật thể nào; trong JSON cũng không có bbox hay polygon.
3. Danh sách tên lớp **đi kèm taxonomy/checkpoint** (ImageNet-1K cho `yolo11n-cls.pt`). Model chỉ chọn trong danh sách có sẵn, không tự nghĩ ra lớp mới.
4. Cùng một `class_name` có thể có `class_id` khác nhau ở taxonomy khác (hoặc cùng ID nhưng khác tên). Ghi đủ ba trường để nhãn hiểu và đối chiếu được, không bị nhầm giữa các bộ lớp.
5. Guideline cần nói rõ: chọn lớp theo vật/chủ đề **chính** nào (to nhất, nổi bật nhất, hay ở trung tâm), cách xử lý khi hai chủ đề ngang nhau, và khi nào được ghi "không xác định/khác".
6. `score` chỉ là mức "tin" của model, không phải sự thật. Model có thể sai mà vẫn có score cao; ground truth phải do người gán theo guideline.

## 2. Phát hiện vật thể

1. Record (`sample_id = kitchen`): `class_name`=<điền> ; `score`=<điền> ; `bbox_xyxy`=[<x_min>, <y_min>, <x_max>, <y_max>] ; `bbox_width`=<điền> ; `bbox_height`=<điền>
2. `bbox_xyxy` cho hai góc: (x_min, y_min) là góc **trên-trái**, (x_max, y_max) là góc **dưới-phải**, gốc tọa độ ở góc trên-trái ảnh. Chiều rộng = x_max − x_min và chiều cao = y_max − y_min, tính bằng pixel. <điền số cụ thể từ record đã chọn>
3. Số prediction theo ngưỡng: `0.20` → <điền> ; `0.60` → <điền> (lấy từ ô so sánh ngưỡng trong notebook).
4. Hạ ngưỡng thì giữ nhiều hộp hơn, trong đó có thêm hộp thừa hoặc sai. Reviewer phải xem và loại nhiều hộp hơn, kiểm tra từng hộp đúng lớp/đúng vị trí không, nên tốn thời gian kiểm tra hơn.
5. Quy tắc box: box phải ôm sát toàn bộ phần nhìn thấy của vật thể, không chừa khoảng trống thừa và không cắt mất phần vật; mỗi vật một box riêng.
6. Vật bị che hoặc cắt mép: <điền tên vật trên ảnh `kitchen`>. Quy tắc chung (vd. "vẽ box quanh phần nhìn thấy") đưa vào guideline được. Nếu trường hợp chưa có quy tắc rõ (vd. che gần hết) thì **escalation** hỏi Lab Coach, không tự đoán.

## 3. Phân đoạn theo từng đối tượng

1. Record (`sample_id = kitchen`): `instance_id`=<điền> ; `class_name`=<điền> ; `score`=<điền> ; `polygon_point_count`=<điền> ; 3–5 điểm đầu của `polygon_xy`=<điền>
2. Box chỉ cho hình chữ nhật bao quanh. Polygon cho **hình dạng thật** của vật: đường viền, phần lõm, phần nhô ra, và loại được vùng nền nằm trong box.
3. `instance_id` phân biệt **từng vật cụ thể** trong cùng một ảnh (hai cái ly cùng lớp "cup" có hai `instance_id` khác nhau). Nó không phải class ID (không nói vật thuộc lớp nào) và không phải tracking ID (không theo vật qua các khung hình video), chỉ có nghĩa trong một ảnh/output này.
4. Quy tắc biên mask: vẽ biên theo mép thật của vật, đủ điểm để bám đúng đường cong, không lấn sang nền hay vật khác.
5. Chỗ khó: <điền, vd. biên mờ / hai vật chạm nhau / bị che trên ảnh `kitchen`>. Nếu guideline đã có quy tắc cho kiểu này thì làm theo guideline. Nếu chưa có, dùng escalation.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh | <điền, vd. ảnh có nhiều vật nên khó chọn một lớp> | Chọn một lớp theo guideline | Lớp có đúng và nhất quán với guideline không |
| Phát hiện vật thể | Mỗi vật một box + lớp | <điền, vd. vật bị che, box thừa/thiếu ở ngưỡng thấp> | Vẽ box sát vật, gán lớp | Box có sát, đủ vật, đúng lớp không |
| Instance segmentation | Mỗi vật một polygon/mask + lớp + `instance_id` | <điền, vd. biên mờ, hai vật chạm nhau> | Vẽ biên theo mép vật, tách từng vật | Biên có đúng, vật có bị gộp/tách sai không |

## 5. An toàn dữ liệu

1. Chỉ dùng ảnh mẫu của bài lab, không đưa họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi dừng lại và báo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
