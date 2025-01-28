import itertools

class CombinationHyperParameters():
    def __init__(self, a,b,c):
        self.a = a
        self.b = b
        self.c = c
        self.res = []

    def generate(self):
        combinations = []
        for d in [self.a, self.b, self.c]:
            combinations.append(
                [
                    dict(zip(d.keys(), combo))
                    for combo in itertools.product(*d.values())
                ]
            )

        self.res = {
            'full_params':combinations[0],
            'male_params':combinations[1],
            'female_params': combinations[2]
            # for a,b,c in itertools.product(combinations[0], combinations[1], combinations[2])
        }


        return self.res
    
    def get_result(self):
        return self.res
    

