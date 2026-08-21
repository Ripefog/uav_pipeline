# uav_pipeline — hướng dẫn cho Claude

## scp: copy file từ gpu-server-5080 về máy client

Mẫu chuẩn — dùng cái này, đừng tự chế `user@IP`:

```bash
scp -r target-anlnm:"/home/anlnm/UAV/uav_pipeline/output/uav0000086_00000_v_person_p4_mcitrack_l384_guarded*" ./
```

Quy tắc rút ra từ mẫu trên:

- **Luôn dùng alias `target-anlnm`**, không dùng `anlnm@172.27.54.58`. Alias nằm
  trong ssh config phía client (Windows), không có trong ssh config của server.
- **Bọc toàn bộ remote path trong dấu nháy kép** khi có `*` hoặc khoảng trắng.
- `*` (glob) chạy được; `{a,b,c}` (brace expansion) thì **KHÔNG**. OpenSSH >= 9.0
  mặc định chuyển scp sang giao thức SFTP, không gọi remote shell, nên `{...}` bị
  truyền nguyên văn và fail `No such file or directory`. Cần brace thì phải thêm
  cờ `-O` để ép về giao thức SCP cũ — nhưng ưu tiên dùng `*` hoặc liệt kê thẳng.
- Nhiều file ở khác thư mục thì đưa nhiều đường dẫn vào một lệnh, hoặc tách lệnh.

Khi user bảo "đưa lệnh scp" thì **chỉ in lệnh ra, không tự chạy**.

## Video output đang được ghi

`cv2.VideoWriter` chỉ ghi moov atom lúc `release()`. Copy file .mp4 khi job còn
chạy sẽ ra file hỏng không mở được — phải đợi job xong mới scp.
