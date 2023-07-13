from sklearn.preprocessing import LabelEncoder


class IdentityEncoder(LabelEncoder):
    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y
