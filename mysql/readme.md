# python sql 学习

## pymysql
这里用通讯录来学习 mysql 的增删改查  

具体流程：  
```
Python 输入数据
    ↓
pymysql 连接 MySQL
    ↓
cursor.execute() 执行 SQL
    ↓
fetchone/fetchall 取结果
    ↓
commit 保存修改
    ↓
close 关闭连接
```


### 初始化
连接 mysql 需要账户密码。这里用 docker 容器开放 3306 端口进行连接。  
每一次操作 mysql 都要建立一次连接，这是短链接。  
``` python
DB_CONFIG: "DBConfig" = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password",
    "charset": "utf8mb4",
    "database": "contacts_db",
    "cursorclass": DictCursor,  # 返回字典类型结果
}


class DBConfig(TypedDict, total=False):
    host: str
    port: int
    user: str
    password: str
    charset: str
   database: str
    cursorclass: type[Cursor]


def get_connection() -> pymysql.Connection:
    return pymysql.connect(**DB_CONFIG)
```

初始化要首先确定数据库与表的建立，这里数据库名为 contacts_db 表为 contacts。  

``` python
def init_db():
    conn = pymysql.connect(
        host="localhost", port=3306, user="root", password="password", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cursor:
            _ = cursor.execute(
                "CREATE DATABASE IF NOT EXISTS contacts_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
    finally:
        conn.close()
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            _ = cursor.execute("""
                CREATE TABLE IF NOT EXISTS contacts(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(50) NOT NULL,
                    phone VARCHAR(20) NOT NULL UNIQUE,
                    email VARCHAR(50),
                    address VARCHAR(100),
                    group_name VARCHAR(20)
                )
            """)
        conn.commit()
    except Exception as e:
        print(f"创建表失败: {e}")
    finally:
        conn.close()
```

每一次建立后创建 cursor 游标对象进行操作。`cursor.execute()` 执行 SQL 语句操作数据库，这里仅将改动记录在内存的回滚日志和重做日志，其他连接者是看不到数据变化的。之后需要运行 `conn.commit()` 刷新日志、标记事物提交、释放锁并通知。    

上述就已经执行了 2 条 SQL 语句。  
``` sql
CREATE DATABASE IF NOT EXISTS contacts_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
```
判断 contacts_db 数据库如果不存在就创建，并且字符编码为 utf8。  

``` sql
CREATE TABLE IF NOT EXISTS contacts(
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    phone VARCHAR(20) NOT NULL UNIQUE,
    email VARCHAR(50),
    address VARCHAR(100),
    group_name VARCHAR(20)
)
```
这个语句用于创建 contacts 的数据表，如果该表在数据库中还不存在的话。  
这几行都在创建每一个列。有数据类型：`INT`、`VARCHAR`。还有限定 `NOT NULL`、`UNIQUE`、`AUTO_INCREMENT`、`PRIMARY KEY` ……  

这样就确定好了数据表的数据项，确定了之后增删改查的基本操作。  

### 增加联系人
pymysql 的 exectue 两个参数： `query` 和 `args`。args 会填到 query 里的 %s 占位符位置。这里 pymysql 把参数传给 MYSQL 而非简单的字符串替换，这里是为了防止 SQL 注入攻击。  

``` python
def add_contact():
    """
    添加联系人
    """
    name = input("姓名:").strip()
    phone = input("电话:").strip()
    email = input("邮箱:").strip()
    address = input("地址:").strip()
    group = input("分组:").strip()

    if not name or not phone:
        print("姓名和电话不能为空。")
        return

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO contacts(name, phone, email, address, group_name) VALUES(%s, %s, %s, %s, %s)"
            cursor.execute(
                sql, (name, phone, email or None, address or None, group or None)
            )
        conn.commit()
        print("添加成功")
    except pymysql.err.IntegrityError as e:
        print(f"添加失败，可能电话重复或数据不合法：{e}")
    except Exception as e:
        print(f"错误：{e}")
    finally:
        conn.close()
```

添加联系人用的 SQL 语句是:
```sql
INSERT INTO contacts(name, phone, email, address, group_name) VALUES(%s, %s, %s, %s, %s)
```

`INSERT INTO contacts` 往 contacts 表插入一条数据。后面加上括号对应列名，这样与 VALUES() 一一对应。SQL 语句在这里仅这样。  

### 显示联系人
``` python
def show_all():
    """
    显示所有联系人
    """
    conn = get_connection()
    try:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute("SELECT * FROM contacts ORDER BY id")
            results = cursor.fetchall()
            if not results:
                print("通讯录为空")
                return
            print("所有联系人:")
            for row in results:
                print(
                    f"ID: {row['id']}, 姓名: {row['name']}, 电话: {row['phone']}, "
                    f"邮箱: {row.get('email', '')}, 地址: {row.get('address', '')}, 分组: {row.get('group_name', '')}"
                )
    finally:
        conn.close()
```

sql 语句如下：  
```python
SELECT * FROM contacts ORDER BY id
```
`SELECT * FROM contacts` 查询所有列  
`ORDER BY id` 按照 id 升序排序。  默认其实是 `ORDER BY id ASC`，ASC 是升序，DESC 是降序。  
用 `cursor.fetchall()` 将所有输出结果取出。因为前面的 cursor 对象建立的时候确定类型是 `DictCursor` 所以数据类型是字典。  
之后就是操纵 python 的字典打印数据。  

### 按对应数据搜索联系人
``` python
def search_contact():
    """
    按姓名或电话搜索
    """
    keyword = input("输入姓名或者电话的关键字：").strip()
    if not keyword:
        print("关键字不能为空")
        return

    conn = get_connection()
    try:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM contacts WHERE name LIKE %s OR phone LIKE %s",
                (f"%{keyword}%", f"%{keyword}%"),
            )
        results = cursor.fetchall()
        if results:
            print("搜索结果：")
            for row in results:
                print(
                    f"ID: {row['id']}, 姓名: {row['name']}, 电话: {row['phone']}, "
                    f"邮箱: {row.get('email', '')}, 地址: {row.get('address', '')}, 分组: {row.get('group_name', '')}"
                )
        else:
            print("没有找到匹配的联系人。")
    finally:
        conn.close()
```
这里的 SQL 语句也是只有一句：  
``` python
cursor.execute(
    "SELECT * FROM contacts WHERE name LIKE %s OR phone LIKE %s",
    (f"%{keyword}%", f"%{keyword}%"),
)
```
`WHERE` 表示查询条件，`name LIKE %s` 表示 name 字段要匹配某种模式，%s 是 pymysql 的参数占位符。 `OR` 是或者。  
参数是 `%{keyword}%` 这类，%% 是 SQL LIKE 里的通配符。  

有如下写法：  
| 写法     | 含义           |
|----------|----------------|
| `"张%"`  | 以张为开头     |
| `"%张"`  | 以张为结尾     |
| `"%张%"` | 包含张         |
| `"张"`   | 必须完全等于张 |


### 修改联系人
``` python
def modify_contact():
    """
    修改联系人
    """
    contact_id = input("输入要修改联系人的 id:").strip()
    if not contact_id.isdigit():
        print("ID 必须为数字")
        return

    conn = get_connection()
    try:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute("SELECT * FROM contacts WHERE id = %s", (int(contact_id),))

            contact = cursor.fetchone()
            if not contact:
                print(f"未找到 ID 为 {contact_id} 的联系人。")
                return

            print("请直接回车保留原值，输入新值则修改。")
            new_name = input(f"姓名 ({contact['name']}): ").strip()
            new_phone = input(f"电话 ({contact['phone']}): ").strip()
            new_email = input(f"邮箱 ({contact.get('email', '')}): ").strip()
            new_address = input(f"地址 ({contact.get('address', '')}): ").strip()
            new_group = input(f"分组 ({contact.get('group_name', '')}): ").strip()

            updates = {}
            if new_name:
                updates["name"] = new_name
            if new_phone:
                updates["phone"] = new_phone
            if new_email:
                updates["email"] = new_email
            if new_address:
                updates["address"] = new_address
            if new_group:
                updates["group_name"] = new_group

            if not updates:
                print("没有修改任何字段。")
                return

            # 把 updates 字典里的每个 key 拼成 字段名=%s 的占位符形式，用逗号连接
            set_clause = ", ".join(f"{k}=%s" for k in updates.keys())
            # 取出所有要更新的值，做成列表
            values = list(updates.values())
            # 把 WHERE 条件用到的 id 追加到值列表末尾，保证与 sql 占位符一一对应
            values.append(int(contact_id))
            sql = f"UPDATE contacts SET {set_clause} WHERE id=%s"
            cursor.execute(sql, values)
        conn.commit()
        print("修改成功")
    finally:
        conn.close()
```

修改就略有复杂度。先找到对应 ID 的一行，然后修改这一行。  
``` python
cursor.execute("SELECT * FROM contacts WHERE id = %s", (int(contact_id),))
```

这里就是直接按 id 值查找对应数据，下面的 `fetchone` 便仅抓取一个。  
之后询问修改，是提取数据后覆盖到原值，不过有匹配列字段进行覆盖。  
使用 `UPDATE`。  
``` python
sql = f"UPDATE contacts SET {set_clause} WHERE id=%s"
```
最终生成的其实是类似这样的 sql 语句：
``` sql
UPDATE contacts SET name=%s, phone=%s WHERE id=%s
```
这样仅匹配需要修改的字段提升效率。  


## 删除联系人

``` python
def delete_contact():
    contact_id = input("请输入要删除的联系人 ID：").strip()
    if not contact_id.isdigit():
        print("ID 必须为数字")
        return
    conn = get_connection()
    try:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute("DELETE FROM contacts WHERE id = %s", (int(contact_id),))
            if cursor.rowcount == 0:
                print("该 ID 不存在，删除失败")
            else:
                conn.commit()
                print("删除成功")

    finally:
        conn.close()
```

仅查找对应 id 进行删除。  
``` sql
DELETE FROM contacts WHERE id = %s
```


