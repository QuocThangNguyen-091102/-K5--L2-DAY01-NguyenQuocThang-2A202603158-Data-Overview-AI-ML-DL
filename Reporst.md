# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh, vì record phân loại không có `bbox_xyxy` hay `polygon_xy` nên không chỉ vào vị trí hay vật thể nào, chỉ có một nhãn `cab` kèm hạng và điểm cho toàn ảnh.

3. Class list đi kèm taxonomy/checkpoint chứ model không tự nghĩ ra, vì `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp cố định của taxonomy đó, không thể dự đoán lớp nằm ngoài danh sách.

4. `class_id` là số máy dùng, ổn định và không bị nhầm chính tả, `class_name` giúp người đọc hiểu nhanh, `taxonomy_name` cho biết đang dùng bảng lớp nào vì cùng một ID có thể mang nghĩa khác ở taxonomy khác (ví dụ ID 468 là `cab` ở ImageNet-1K nhưng COCO-80 không có ID này), thiếu taxonomy thì không giải mã ID chắc chắn được.

5. Guideline cần nói rõ tiêu chí chọn (chủ thể chính, vật chiếm diện tích lớn nhất hay chủ đề tổng thể), quy tắc ưu tiên khi nhiều ứng viên ngang nhau (như `cab`, `minibus`, `police_van` ở ảnh traffic), cách xử lý ca mơ hồ (có nhãn "khác/không rõ" hay không, khi nào escalation hỏi Lab Coach thay vì tự đoán), và có ví dụ minh họa (đúng, sai, ca biên).

6. `score` chỉ là độ tin cậy của model để xếp hạng hoặc lọc prediction, không phải điểm chất lượng của nhãn, và score cao không đảm bảo đúng (ví dụ ảnh kitchen ra `gong` với 0.42), nên ground truth phải do người gán nhãn theo guideline tạo ra và reviewer kiểm tra, nếu lấy prediction làm đáp án thì model chỉ học lại lỗi của chính nó.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Góc trên-trái của hộp nằm ở (385.33, 69.24), tức cách mép trái ảnh khoảng 385 pixel và cách mép trên khoảng 69 pixel, góc dưới-phải nằm ở (498.92, 348.92), hộp rộng khoảng 113.58 pixel và cao khoảng 279.68 pixel, gốc (0, 0) ở góc trên-trái ảnh, và hộp bao quanh người đứng quay lưng ở giữa bếp.

3. Ngưỡng 0.35 có 11 prediction (person, bowl, oven, cup...), ngưỡng 0.60 có 6 prediction (2 person, 2 bowl, 2 oven).

4. Hạ ngưỡng thì giữ lại nhiều hộp hơn nên reviewer phải kiểm tra thêm từng hộp thừa, loại hộp sai (nhầm lớp, hộp nhận nhầm vật không tồn tại, hộp trùng lặp), sửa hộp lệch và tìm xem hộp nào còn thiếu, nên khối lượng kiểm tra và rủi ro bỏ sót lỗi đều tăng.

5. Vẽ một hộp chữ nhật cho mỗi vật thể, ôm sát toàn bộ phần nhìn thấy của vật (cạnh hộp chạm sát điểm ngoài cùng, không chừa viền thừa, không cắt mất phần vật), không gộp hai vật thành một hộp và không để hộp tràn ra ngoài ảnh.

6. Vật bị cắt mép: người (`person`, score 0.61) ở mép trái ảnh có `x_min`=0.12 nên chỉ thấy một phần cơ thể, lại nằm lẫn với các tô bowl phía trước, tôi sẽ viết thành quy tắc trong guideline ("vật bị che khuất hoặc cắt mép thì chỉ vẽ hộp ôm phần nhìn thấy, không đoán phần bị che"), vì tình huống này lặp lại ở rất nhiều ảnh nên cần một cách làm thống nhất cho mọi annotator, còn những ca mà không xác định được đó là vật gì thì mới escalation hỏi Lab Coach thay vì tự đoán nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. Box chỉ cho biết vật nằm trong một hình chữ nhật (ở kitchen-001 là [385.45, 66.44, 498.02, 348.58]) nên chứa cả nền thừa ở các góc, còn polygon cho biết hình dạng thật của vật: đầu tròn, vai, hai tay buông, hai chân đứng hơi tách ra, nhờ vậy biết chính xác pixel nào thuộc vật và pixel nào là nền, và có thể tính diện tích hay tách hai vật chồng lấn nhau, những việc mà box không làm được.

3. `instance_id` (dạng `kitchen-001`) dùng để phân biệt từng vật thể riêng lẻ trong output của bài lab, nên hai vật cùng lớp như `kitchen-002` và `kitchen-003` (đều là bowl) vẫn có mã và polygon riêng, còn câu "không phải class ID, không phải tracking ID" nghĩa là nó không cho biết vật thuộc loại gì (class ID như 45 = bowl, 0 = person) và cũng không dùng để theo dõi cùng một vật qua nhiều ảnh hay nhiều khung hình video, nó chỉ là số thứ tự đánh nhãn trong một ảnh của bộ output này.

4. Đặt các điểm polygon bám sát đường biên ngoài của phần nhìn thấy của vật, đủ dày ở chỗ cong để mask không bị cắt góc hay tràn ra nền, không đưa vật khác hay nền vào trong mask, và với vật bị che thì chỉ vẽ phần nhìn thấy, không đoán phần bị che.

5. Chỗ khó là bàn (`dining table`, kitchen-005, score 0.58) bị nhiều bowl, cup và các vật khác đặt chồng lên nên mask của bàn có tới 598 điểm với biên rất lộn xộn, đồng thời các bowl (kitchen-002, 003, 006) chạm và chồng lên nhau ở góc trái ảnh nên khó biết biên của từng bowl nằm ở đâu, tôi sẽ dùng guideline cho quy tắc chung (vẽ phần nhìn thấy, vật đặt trên bàn thì khoét ra khỏi mask bàn, hai vật chạm nhau thì mỗi vật một polygon riêng theo đường tiếp giáp nhìn thấy), vì tình huống này lặp lại ở rất nhiều ảnh và cần cách làm thống nhất giữa các annotator, còn các ca không thể nhìn ra biên (bị che gần hết, quá mờ hoặc không xác định được là vật gì) thì mới escalation hỏi Lab Coach thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho cả bức ảnh (đơn vị nhãn là một ảnh), lấy từ taxonomy cố định, gồm `class_id`, `class_name`, `taxonomy_name`. | Ảnh `traffic` có top-1 `cab` (0.51) nhưng `minibus` (0.16) và `police_van` (0.09) cũng hợp lý; ảnh `kitchen` ra `gong` (0.42) dù là căn bếp vì ImageNet-1K không có lớp phù hợp. | Chọn đúng một lớp theo tiêu chí trong guideline (chủ thể chính hoặc chủ đề tổng thể), ca mơ hồ thì hỏi Lab Coach, không tự đoán. | Kiểm tra lớp có nằm trong taxonomy không, có đúng tiêu chí chọn không, ca mơ hồ có được xử lý nhất quán không, không dựa vào model score. |
| Phát hiện vật thể | Mỗi vật thể có một lớp và một hộp `bbox_xyxy` = [x_min, y_min, x_max, y_max] theo pixel, gốc (0, 0) ở góc trên-trái. | Ảnh `kitchen` có `person` (score 0.61) bị cắt ở mép trái (`x_min`=0.12), lẫn với nhiều `bowl` chồng nhau (score 0.38 đến 0.72); số hộp đổi theo ngưỡng: 11 hộp ở 0.35 và 6 hộp ở 0.60. | Vẽ một hộp ôm sát phần nhìn thấy của mỗi vật, không gộp hai vật thành một hộp, không bỏ sót vật nhỏ hay vật bị che, escalation khi không xác định được vật. | Kiểm tra sai lớp, hộp lệch hoặc quá rộng, hộp trùng lặp, vật bị bỏ sót, hộp tràn ra ngoài ảnh. |
| Instance segmentation | Mỗi instance có một lớp, một hộp và một đa giác `polygon_xy` bám biên vật, kèm `instance_id` riêng; hai vật cùng lớp vẫn có polygon riêng. | Ảnh `kitchen` có `dining table` (kitchen-005, 598 điểm) bị bowl, cup và vật khác đè lên nên biên lộn xộn; các `bowl` (kitchen-002, 003, 006) chồng nhau ở góc trái; `person` (kitchen-009) bị cắt ở mép ảnh. | Vẽ polygon bám sát biên phần nhìn thấy, vật đặt trên bàn thì khoét ra khỏi mask của bàn, hai vật chạm nhau thì mỗi vật một polygon, escalation khi không thấy rõ biên. | Kiểm tra mask có tràn ra nền hoặc cắt mất phần vật không, polygon có chồng lấn sai giữa các instance không, lớp có đúng không, `instance_id` có duy nhất không. |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, không tải ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab/GitHub công khai, và không ghi họ tên, MSSV vào báo cáo hay output.

2. Tôi sẽ dừng ngay, không upload hay commit dữ liệu đó, và báo cho Lab Coach (hoặc GV/Lab Lead); nếu lỡ đã push thì báo ngay để xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

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
