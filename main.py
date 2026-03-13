import sqlite3
import os

# 1. 定义数据库路径，完美利用咱们刚才建的 data 文件夹
db_path = os.path.join("data", "test.db")

# 2. 连接数据库（如果没有这个文件，Python 会自动帮你创建一个）
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 3. 创建一张测试用的数据表
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY,
        nickname TEXT,
        major TEXT
    )
"""
)

# 插入一条测试数据（顺便清空一下表防止重复插入报错）
cursor.execute("DELETE FROM students")
cursor.execute(
    "INSERT INTO students (nickname, major) VALUES ('Kinoko', 'Computer Science')"
)
conn.commit()

# ==========================================
# 4. Text2SQL 核心第一步：提取数据库的语义信息 (Schema)
# ==========================================
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
schemas = cursor.fetchall()

print("🌟 恭喜！Text2SQL 系统核心模块启动成功！")
print(f"📁 数据库文件已成功生成于: {db_path}")
print("-" * 40)
print("🔍 提取到的数据库语义信息 (Schema):")
for schema in schemas:
    print(schema[0])

conn.close()
