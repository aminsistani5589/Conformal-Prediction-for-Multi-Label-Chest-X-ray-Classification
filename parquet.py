import pandas as pd

# فرض کن اسم فایل تو 'chexnet_data.parquet' هست
# مسیر فایلت رو جایگزین کن
file_path = 'D:\\valid-00000-of-00010.parquet' 

# لود کردن دیتاست
try:
    df = pd.read_parquet(file_path)
    
    # یه نگاه سریع بهش بندازیم ببینیم تو شکمش چی داره
    print("ساختار دیتاست:")
    print(df.head())
    print("\nاطلاعات ستون‌ها:")
    print(df.info())

except Exception as e:
    print(f"اوپس! یه جای کار می‌لنگه: {e}")
