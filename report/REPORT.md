# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** <điền>

**Runtime Colab:** <CPU hoặc GPU>

**Python / PyTorch / Ultralytics:** <điền Python> / <điền PyTorch> / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

1. Record hạng 1 (`sample_id = traffic`): `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**. Phân loại ảnh gán một nhãn cho toàn bộ ảnh, không chỉ ra vị trí hay ranh giới của vật thể nào; trong JSON cũng không có `bbox_xyxy` hay `polygon_xy`, chỉ có một nhãn `cab` kèm hạng và điểm cho toàn ảnh.
3. Danh sách tên lớp **đi kèm taxonomy/checkpoint** (ImageNet-1K cho `yolo11n-cls.pt`). Model được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp cố định của taxonomy đó, không tự nghĩ ra lớp mới hay dự đoán lớp nằm ngoài danh sách.
4. `class_id` là số máy dùng, ổn định và không bị nhầm chính tả; `class_name` giúp người đọc hiểu nhanh; `taxonomy_name` cho biết đang dùng bảng lớp nào. Cùng một ID có thể mang nghĩa khác ở taxonomy khác (ví dụ ID 468 là `cab` ở ImageNet-1K nhưng COCO-80 không có ID này), nên thiếu taxonomy thì không giải mã ID chắc chắn được. Ghi đủ ba trường để nhãn hiểu và đối chiếu được, không bị nhầm giữa các bộ lớp.
5. Guideline cần nói rõ: chọn lớp theo vật/chủ đề **chính** nào (chủ thể chính, vật chiếm diện tích lớn nhất hay chủ đề tổng thể), quy tắc ưu tiên khi nhiều ứng viên ngang nhau (như `cab`, `minibus`, `police_van` ở ảnh traffic), khi nào được ghi "không xác định/khác", khi nào escalation hỏi Lab Coach thay vì tự đoán, và có ví dụ minh họa (đúng, sai, ca biên).
6. `score` chỉ là mức "tin" của model để xếp hạng hoặc lọc prediction, không phải sự thật hay điểm chất lượng nhãn. Model có thể sai mà vẫn có score cao (ví dụ ảnh kitchen ra `gong` với 0.42); ground truth phải do người gán theo guideline và reviewer kiểm tra, nếu lấy prediction làm đáp án thì model chỉ học lại lỗi của chính nó.

## 2. Phát hiện vật thể

1. Record (`sample_id = kitchen`): `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. `bbox_xyxy` cho hai góc: (x_min, y_min) là góc **trên-trái**, (x_max, y_max) là góc **dưới-phải**, gốc tọa độ ở góc trên-trái ảnh. Chiều rộng = x_max − x_min và chiều cao = y_max − y_min, tính bằng pixel. Với record đã chọn: góc trên-trái ở (385.33, 69.24), tức cách mép trái ảnh khoảng 385 pixel và cách mép trên khoảng 69 pixel; góc dưới-phải ở (498.92, 348.92); chiều rộng = 498.92 − 385.33 = 113.58 pixel, chiều cao = 348.92 − 69.24 = 279.68 pixel. Hộp bao quanh người đứng quay lưng ở giữa bếp.
3. Số prediction theo ngưỡng: `0.20` → <điền> ; `0.35` → 11 (person, bowl, oven, cup...) ; `0.60` → 6 (2 person, 2 bowl, 2 oven) (lấy từ ô so sánh ngưỡng trong notebook).
4. Hạ ngưỡng thì giữ nhiều hộp hơn, trong đó có thêm hộp thừa hoặc sai (nhầm lớp, nhận nhầm vật không tồn tại, hộp trùng lặp). Reviewer phải xem và loại nhiều hộp hơn, sửa hộp lệch, kiểm tra từng hộp đúng lớp/đúng vị trí không và tìm hộp còn thiếu, nên tốn thời gian kiểm tra hơn và rủi ro bỏ sót lỗi cũng tăng.
5. Quy tắc box: box phải ôm sát toàn bộ phần nhìn thấy của vật thể (cạnh hộp chạm sát điểm ngoài cùng), không chừa khoảng trống thừa và không cắt mất phần vật; mỗi vật một box riêng, không gộp hai vật vào một box và không để box tràn ra ngoài ảnh.
6. Vật bị che hoặc cắt mép: người (`person`, score 0.61) ở mép trái ảnh `kitchen` có `x_min`=0.12 nên chỉ thấy một phần cơ thể, lại nằm lẫn với các tô `bowl` phía trước. Quy tắc chung ("vật bị che khuất hoặc cắt mép thì chỉ vẽ box quanh phần nhìn thấy, không đoán phần bị che") đưa vào guideline được, vì tình huống này lặp lại ở rất nhiều ảnh nên cần một cách làm thống nhất cho mọi annotator. Nếu trường hợp chưa có quy tắc rõ (vd. che gần hết, không xác định được là vật gì) thì **escalation** hỏi Lab Coach, không tự đoán.

## 3. Phân đoạn theo từng đối tượng

1. Record (`sample_id = kitchen`): `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]
2. Box chỉ cho hình chữ nhật bao quanh (ở kitchen-001 là [385.45, 66.44, 498.02, 348.58]) nên chứa cả nền thừa ở các góc. Polygon cho **hình dạng thật** của vật: đầu tròn, vai, hai tay buông, hai chân đứng hơi tách ra; nhờ vậy biết chính xác pixel nào thuộc vật và pixel nào là nền, loại được vùng nền nằm trong box, tính được diện tích và tách được hai vật chồng lấn nhau, những việc mà box không làm được.
3. `instance_id` phân biệt **từng vật cụ thể** trong cùng một ảnh: hai vật cùng lớp như `kitchen-002` và `kitchen-003` (đều là `bowl`) vẫn có mã và polygon riêng. Nó không phải class ID (không nói vật thuộc lớp nào, như 45 = bowl, 0 = person) và không phải tracking ID (không theo cùng một vật qua nhiều ảnh hay khung hình video), chỉ là số thứ tự đánh nhãn có nghĩa trong một ảnh/output này.
4. Quy tắc biên mask: đặt các điểm polygon bám sát mép thật của phần nhìn thấy của vật, đủ dày ở chỗ cong để mask không bị cắt góc, không lấn sang nền hay vật khác; vật bị che thì chỉ vẽ phần nhìn thấy, không đoán phần bị che.
5. Chỗ khó: bàn (`dining table`, kitchen-005, score 0.58) bị nhiều bowl, cup và vật khác đặt chồng lên nên mask có tới 598 điểm với biên rất lộn xộn; các bowl (kitchen-002, 003, 006) chạm và chồng lên nhau ở góc trái ảnh nên khó biết biên của từng bowl nằm ở đâu. Guideline nên có quy tắc chung cho kiểu này (vẽ phần nhìn thấy, vật đặt trên bàn thì khoét ra khỏi mask bàn, hai vật chạm nhau thì mỗi vật một polygon riêng theo đường tiếp giáp nhìn thấy) và annotator làm theo guideline. Ca không thể nhìn ra biên (bị che gần hết, quá mờ hoặc không xác định được là vật gì) thì dùng escalation hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh (đơn vị nhãn là một ảnh), lấy từ taxonomy cố định, gồm `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có top-1 `cab` (0.51) nhưng `minibus` (0.16) và `police_van` (0.09) cũng hợp lý; ảnh `kitchen` ra `gong` (0.42) dù là căn bếp vì ImageNet-1K không có lớp phù hợp | Chọn một lớp theo guideline (chủ thể chính hoặc chủ đề tổng thể), ca mơ hồ thì hỏi Lab Coach, không tự đoán | Lớp có nằm trong taxonomy, có đúng và nhất quán với guideline không, không dựa vào model score |
| Phát hiện vật thể | Mỗi vật một box + lớp, `bbox_xyxy` = [x_min, y_min, x_max, y_max] theo pixel, gốc (0, 0) ở góc trên-trái | Ảnh `kitchen` có `person` (score 0.61) bị cắt ở mép trái (`x_min`=0.12), lẫn với nhiều `bowl` chồng nhau (score 0.38 đến 0.72); số hộp đổi theo ngưỡng: 11 hộp ở 0.35 và 6 hộp ở 0.60 | Vẽ box sát phần nhìn thấy của vật, gán lớp, không gộp hai vật, không bỏ sót vật nhỏ hay vật bị che, escalation khi không xác định được vật | Box có sát, đủ vật, đúng lớp không; có hộp lệch, trùng lặp, tràn ra ngoài ảnh hay vật bị bỏ sót không |
| Instance segmentation | Mỗi vật một polygon/mask + lớp + `instance_id`; hai vật cùng lớp vẫn có polygon riêng | Ảnh `kitchen` có `dining table` (kitchen-005, 598 điểm) bị bowl, cup đè lên nên biên lộn xộn; các `bowl` (kitchen-002, 003, 006) chồng nhau ở góc trái; `person` (kitchen-009) bị cắt ở mép ảnh | Vẽ biên theo mép phần nhìn thấy, tách từng vật, vật đặt trên bàn thì khoét khỏi mask bàn, escalation khi không thấy rõ biên | Biên có đúng, mask có tràn nền hay cắt mất vật không, vật có bị gộp/tách sai không, lớp đúng không, `instance_id` có duy nhất không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai, và không đưa họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi, tôi dừng lại, không upload hay commit dữ liệu đó, và báo cho Lab Coach (hoặc GV/Lab Lead); nếu lỡ đã push thì báo ngay để xử lý.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
