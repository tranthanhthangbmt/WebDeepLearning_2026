/**
 * ====================================================================
 * MÃ GOOGLE APPS SCRIPT: TỰ ĐỘNG LƯU ĐIỂM BÀI KIỂM TRA MÔN HỌC SÂU (DEEP LEARNING)
 * Học phần: Học Sâu (Deep Learning) - Đại học Đông Á
 * Google Sheet liên kết: https://docs.google.com/spreadsheets/d/1jE5pqvoSDxC_NG6VmGULdFj6_Kj5miL1mInSCx4VrQc/edit?gid=0#gid=0
 * ("Điểm Giữa kỳ AI Kế toán_15.10.2026")
 * ====================================================================
 * 
 * HƯỚNG DẪN CÀI ĐẶT NHANH (MẤT 2 PHÚT NẾU CẦN TẠO SHEET MỚI):
 * --------------------------------------------------------------------
 * 1. Mở trang tính Google Sheet tại:
 *    https://docs.google.com/spreadsheets/d/1jE5pqvoSDxC_NG6VmGULdFj6_Kj5miL1mInSCx4VrQc/edit?gid=0#gid=0
 * 2. Trên thanh Menu của Google Sheet, chọn:
 *    -> Tiện ích mở rộng (Extensions) -> Apps Script.
 * 3. Dán toàn bộ đoạn mã này vào trình soạn thảo code.
 * 4. Bấm nút "Lưu" (biểu tượng đĩa mềm 💾 hoặc Ctrl + S).
 * 5. Bấm vào nút "Triển khai" (Deploy) màu xanh ở góc trên bên phải:
 *    -> Chọn "Tùy chọn triển khai mới" (New deployment).
 *    -> Bấm vào biểu tượng Bánh răng (⚙️) bên cạnh "Chọn loại", chọn: "Ứng dụng web" (Web App).
 *    -> Mô tả: "Lưu điểm thi trắc nghiệm giữa kỳ môn Học Sâu".
 *    -> Thực thi dưới dạng (Execute as): "Tôi" (Me).
 *    -> Ai có quyền truy cập (Who has access): "Bất kỳ ai" (Anyone) — RẤT QUAN TRỌNG!
 * 6. Bấm nút "Triển khai" (Deploy) và cấp quyền truy cập tài khoản Google.
 * 7. Sao chép "URL ứng dụng web" (Web App URL có dạng https://script.google.com/macros/s/.../exec).
 *    URL hiện tại đã cấu hình sẵn trong hệ thống:
 *    https://script.google.com/macros/s/AKfycby2cPdyEuXF8Hyda1yiqgF5zFo3Ya5qhMjua2YUkyzcFZ-GdhFRGT19WgjhEgqmi-ziAA/exec
 */

function doPost(e) {
  var lock = LockService.getScriptLock();
  // Khóa script tối đa 15 giây để xếp hàng các sinh viên nộp bài cùng lúc, tránh ghi đè dòng
  lock.tryLock(15000);

  try {
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sheet = ss.getActiveSheet();

    // Tự động tạo dòng tiêu đề nếu trang tính còn mới
    if (sheet.getLastRow() === 0) {
      sheet.appendRow([
        "Thời gian nộp", 
        "Họ và tên sinh viên", 
        "Mã sinh viên (MSSV)", 
        "Lớp sinh hoạt", 
        "Lần thi",
        "Điểm (Thang 10)", 
        "Số câu đúng", 
        "Tổng số câu", 
        "Thời gian làm bài", 
        "Tỷ lệ hoàn thành",
        "Số lần rời màn hình (Vi phạm)"
      ]);
      
      // Định dạng dòng tiêu đề xanh dương công nghệ Học Sâu
      var headerRange = sheet.getRange(1, 1, 1, 11);
      headerRange.setFontWeight("bold");
      headerRange.setBackground("#1a73e8");
      headerRange.setFontColor("#ffffff");
      headerRange.setHorizontalAlignment("center");
      headerRange.setVerticalAlignment("middle");
      sheet.setRowHeight(1, 35);
      sheet.setFrozenRows(1);
    }

    // Đọc dữ liệu JSON gửi từ ứng dụng web
    var data = {};
    if (e.postData && e.postData.contents) {
      data = JSON.parse(e.postData.contents);
    }

    var row = [
      data.submitTime || Utilities.formatDate(new Date(), "GMT+7", "dd/MM/yyyy HH:mm:ss"),
      data.fullName || "Chưa có tên",
      data.studentId || "Chưa có MSSV",
      data.className || "Chưa có lớp",
      data.attemptNumber !== undefined ? "Lần " + data.attemptNumber : "Lần 1",
      data.score10 !== undefined ? Number(data.score10) : 0,
      data.correctCount !== undefined ? Number(data.correctCount) : 0,
      data.totalQuestions || 90,
      data.timeSpent || "",
      data.summaryText || "",
      data.tabSwitches !== undefined ? Number(data.tabSwitches) : 0
    ];

    sheet.appendRow(row);
    
    // Canh chỉnh giao diện các ô vừa thêm
    var lastRow = sheet.getLastRow();
    sheet.setRowHeight(lastRow, 26);
    sheet.getRange(lastRow, 1).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 3).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 4).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 5).setHorizontalAlignment("center");
    sheet.getRange(lastRow, 6).setHorizontalAlignment("center").setFontWeight("bold").setFontColor("#1a73e8");
    sheet.getRange(lastRow, 7, 1, 5).setHorizontalAlignment("center");
    if (data.tabSwitches && Number(data.tabSwitches) > 0) {
      sheet.getRange(lastRow, 11).setFontColor("#dc2626").setFontWeight("bold");
    } else {
      sheet.getRange(lastRow, 11).setFontColor("#16a34a");
    }

    return ContentService.createTextOutput(JSON.stringify({ 
      "status": "success", 
      "message": "Đã lưu kết quả thi môn Học Sâu (Deep Learning) thành công vào Google Sheet!",
      "row": lastRow
    })).setMimeType(ContentService.MimeType.JSON);

  } catch (err) {
    return ContentService.createTextOutput(JSON.stringify({ 
      "status": "error", 
      "message": err.toString() 
    })).setMimeType(ContentService.MimeType.JSON);

  } finally {
    lock.releaseLock();
  }
}

function doGet(e) {
  return ContentService.createTextOutput("Hệ thống Web App chấm điểm môn Học Sâu (Đại học Đông Á) đang hoạt động bình thường!");
}
