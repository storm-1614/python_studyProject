from typing import TypedDict

import pymysql
from pymysql.cursors import Cursor, DictCursor


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
    except Exception as e:
        print(f"创建表失败: {e}")
    finally:
        conn.close()


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


def main_menu():
    """主菜单"""
    while True:
        print("\n===== 通讯录管理系统 =====")
        print("1. 添加联系人")
        print("2. 查看所有联系人")
        print("3. 搜索联系人")
        print("4. 修改联系人")
        print("5. 删除联系人")
        print("0. 退出")
        choice = input("请输入选项: ").strip()

        if choice == "1":
            add_contact()
        elif choice == "2":
            show_all()
        elif choice == "3":
            search_contact()
        elif choice == "4":
            modify_contact()
        elif choice == "5":
            delete_contact()
        elif choice == "0":
            print("再见！")
            break
        else:
            print("无效选项，请重新输入。")


if __name__ == "__main__":
    init_db()
    main_menu()
