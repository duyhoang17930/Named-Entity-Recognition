import pandas as pd
from newspaper import Article
import nltk
from googlenewsdecoder import new_decoderv1
from DrissionPage import ChromiumPage, ChromiumOptions
import time
import os

# Tải dữ liệu chia câu
nltk.download('punkt')
nltk.download('punkt_tab')

def get_real_url(google_url):
    if 'news.google.com' in google_url:
        try:
            decoded = new_decoderv1(google_url, interval=1)
            if decoded.get('status'):
                return decoded['decoded_url']
        except:
            return google_url
    return google_url

def scrape_with_browser(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}")
        return

    # CẤU HÌNH TRÌNH DUYỆT NHƯ NGƯỜI THẬT
    co = ChromiumOptions()
    # Tạm thời tắt headless để vượt qua Reuters. 
    # Nếu muốn ẩn, hãy đổi thành co.set_argument('--headless', 'new')
    # co.set_argument('--headless', 'new') 
    
    page = ChromiumPage(co)

    df = pd.read_csv(input_file) if input_file.endswith('.csv') else pd.read_excel(input_file)
    all_data = []

    print(f"Bắt đầu xử lý {len(df)} link...")

    try:
        for index, row in df.iterrows():
            google_url = row['Link']
            print(f"[{index+1}] Đang xử lý...")

            real_url = get_real_url(google_url)
            print(f"      -> Truy cập: {real_url[:70]}...")

            try:
                page.get(real_url)
                
                # CHỜ ĐỢI VÀ CUỘN TRANG ĐỂ LOAD NỘI DUNG
                time.sleep(2)
                page.scroll.to_half() # Cuộn xuống giữa trang
                time.sleep(1)
                page.scroll.to_bottom() # Cuộn xuống cuối trang
                time.sleep(1)

                # Lấy HTML sau khi đã load hết
                html_content = page.html
                
                # Dùng Newspaper bóc tách
                article = Article(real_url, language='en') 
                article.download(input_html=html_content) 
                article.parse()
                
                # Nếu bóc tách tự động thất bại, thử lấy thủ công qua các thẻ p
                text = article.text
                if len(text) < 200:
                    print("      [!] Newspaper thất bại, đang thử lấy text thủ công...")
                    paragraphs = page.eles('tag:p')
                    text = "\n".join([p.text for p in paragraphs if len(p.text) > 50])

                sentences = nltk.sent_tokenize(text)

                count = 0
                for sent in sentences:
                    sent = sent.strip().replace('\n', ' ')
                    if len(sent) > 40 and not sent.startswith('Copyright'):
                        all_data.append({
                            # 'Source': row['Source'],
                            'Title': row['Title'],
                            # 'Link': real_url,
                            'Sentence': sent
                        })
                        count += 1
                
                if count > 0:
                    print(f"      [OK] Thành công! Lấy được {count} câu.")
                else:
                    print(f"      [!] Thất bại: Không tìm thấy nội dung bài báo.")

            except Exception as e:
                print(f"      [!] Lỗi truy cập: {e}")
            
            # Nghỉ ngắn giữa các bài
            time.sleep(1)

        # Lưu file
        if all_data:
            result_df = pd.DataFrame(all_data)
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n--- HOÀN TẤT: Đã lưu vào {output_file} ---")

    finally:
        page.quit()

# Chạy
scrape_with_browser("raw_data_20260203.csv", "ket_qua_sentence.csv")