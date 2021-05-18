import sqlite3

class MemberModel:
    def __init__(self):
        self.conn = sqlite3.connect('sqlite.db')
    def create(self):
        query = """
            CREATE TABLE IF NOT EXISTS MEMBER(
                USERID VARCHAR(10) PRIMARY KEY,
                PASSWORD VARCHAR(10),
                PHONE VARCHAR(15),
                REGDATE DATE DEFAULT CURRENT_TIMESTAMP
            );
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_many(self):
        data = [
            ('lee', '1', '010-1234-5678'),
            ('kim', '1', '010-1234-5677'),
            ('park', '1', '010-1234-5670')
        ]
        stmt = """
            INSERT INTO MEMBER(USERID, PASSWORD, PHONE) VALUES(?, ?, ?)
        """
        self.conn.executemany(stmt, data)
        self.conn.commit()

    def fetch_one(self):
        cursor = self.conn.execute("SELECT * FROM MEMBER WHERE USERID LIKE 'lee'")
        row = cursor.fetchone()
        print('검색결과 : {}'.format(row))

    def fetch_all(self):
        cursor = self.conn.execute('SELECt * FROM MEMBER')
        rows = cursor.fetchall()
        count = 0
        for i in rows:
            count += 1
        print('총인원 {}'.format(count))

    def login(self, userid, password):
        query = """
            SELECT * FROM MEMBER
            WHERE USERID LIKE ?
                AND PASSWORD LIKE ?
        """
        data = [userid, password]
        cursor = self.conn.execute(query, data)
        row = cursor.fetchone()
        print('로그인 회원정보: {}'.format(row))
        return row