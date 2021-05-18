from members.model import MemberModel
class MemberController:
    def __init__(self):
        self.model = MemberModel()

    def creat_table(self):
        self.model.create()
        self.model.insert_many()
        self.model.fetch_all()

    def login(self, userid, password):
        row = self.model.login(userid, password)
        if row is None:
            view = 'login.html'
        else:
            view = 'index.html'
        return view