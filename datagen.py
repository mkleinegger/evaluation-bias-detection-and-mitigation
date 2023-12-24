import pandas as pd
from scipy.stats import bernoulli, poisson, norm, expon
import random
import string

# ablauf
# data will be generated based on schema
# we use a an own classification to get a classification on values
# we drizzle the whizzle the classification


class DataGen:
    def __init__(self, schema, config, noise, classifier):
        self.schema = schema
        self.config = config
        self.noise = noise
        self.classifier = classifier

    def __repeat_array(self, arr, n):
        m = len(arr)
        repetitions = n // m
        remainder = n % m

        repeated_arr = arr * repetitions + arr[:remainder]
        return repeated_arr

    def __generate_random_text(self, min_length, max_length):
        return "".join(
            random.choice(string.ascii_letters + string.digits + string.punctuation)
            for _ in range(min_length, max_length)
        )

    def generate_data(self, n: int):
        # create pandas dataframe
        cols = {}

        for name, attributes in self.schema.items():
            col = []

            match attributes["type"]:
                case "ratio":
                    min_range, max_range = attributes["range"]
                    col = norm.rvs(loc=0, scale=max_range, size=n)
                case "nominal":
                    col = self.__repeat_array(attributes["values"], n)
                case "text":
                    min_range, max_range = attributes["length_range"]
                    col = [
                        self.__generate_random_text(min_range, max_range)
                        for _ in range(n)
                    ]
                case _:
                    raise ValueError("Invalid type!")

            cols[name] = col

        df = pd.DataFrame(cols)
        df["target"] = df.apply(self.classifier, axis=1)

        return df

    def display_info(self):
        print(f"{self.schema} {self.config} {self.noise} ({self.classifier})")


class DataGenBuilder:
    def __init__(self) -> None:
        self.schema = None
        self.config = None
        self.noise = None
        self.classifier = None

    def set_schema(self, schema: dict) -> "DataGenBuilder":
        self.schema = schema
        return self

    def set_config(self, config) -> "DataGenBuilder":
        self.config = config
        return self

    def set_noise(self, noise: bool) -> "DataGenBuilder":
        self.noise = noise
        return self

    def set_classification(self, classifier: callable) -> "DataGenBuilder":
        self.classifier = classifier
        return self

    def build(self) -> DataGen:
        if (
            self.config is None
            or self.schema is None
            or self.noise is None
            or self.classifier is None
        ):
            raise ValueError("Cannot build DataGen instance. Missing information.")

        return DataGen(self.schema, self.config, self.noise, self.classifier)


builder: DataGenBuilder = (
    DataGenBuilder()
    .set_schema(
        {
            "feature1": {"type": "ratio", "distribution": "normal", "range": (1, 100)},
            "feature2": {
                "type": "nominal",
                "distribution": "normal",
                "values": [1, 2, 3],
            },
            "feature3": {
                "type": "text",
                "distribution": "normal",
                "length_range": (5, 25),
            },
            "feature4": {"type": "ratio", "distribution": "normal", "range": (1, 50)},
        }
    )
    .set_config("test")
    .set_noise(True)
    .set_classification(lambda x: x[0] > 0)
)

dataGen: DataGen = builder.build()
# data = dataGen.generate_data(10_000)
# print(data)
