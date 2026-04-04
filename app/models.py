from flask_login import UserMixin
from firebase_admin import firestore

db = firestore.client()

class User(UserMixin):
    def __init__(self, uid, email, role="patient", name=None, specialization=None, license_number=None):
        self.id = uid
        self.email = email
        self.role = role
        self.name = name
        self.specialization = specialization
        self.license_number = license_number

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "name": self.name,
            "specialization": self.specialization,
            "license_number": self.license_number
        }

    @staticmethod
    def get_by_id(uid):
        try:
            doc = db.collection("users").document(uid).get()
            if not doc.exists:
                return None

            data = doc.to_dict()
            return User(
                uid=uid,
                email=data.get("email"),
                role=data.get("role", "patient"),
                name=data.get("name"),
                specialization=data.get("specialization"),
                license_number=data.get("license_number")
            )
        except Exception as e:
            print(f"[User.get_by_id] Error fetching user {uid}: {e}")
            return None
