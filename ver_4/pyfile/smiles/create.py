groups = {
    "h": ("", 0),
    "cn": ("C#N", 0),
    "no2": ("N(=O)=O", 0),
    "f": ("F", 0),
    "cl": ("Cl", 0),
    "br": ("Br", 0),
    "c6h5": ("C2=CC=CC=C2", 0),
    "ch3": ("C", 1),
    "oh": ("O", 1),
    "cho": ("C(=O)", 1),
    "cooh": ("C(=O)O", 1),
}

base = {
    3: ["C", 1],
    4: ["C", 1],
    5: ["C", 1],
    6: ["C", 1],
}

length = 2


def make_group(base_group, base_num, add_group, tf, i):
    if base_num == 0:
        return base_group, base_num

    if add_group == "" and i == 0:
        base_num = 0
        return base_group, base_num
    elif add_group == "" and i != 0:
        base_num = 0
        return base_group + ")", base_num

    if i == 0:
        base_group += "("

    base_group += add_group

    if (i == length - 1) or (tf == 0):
        base_num = 0
        return base_group + ")", base_num
    return base_group, base_num


for c3 in groups.values():
    base[3] = ["C", 1]
    for i in range(length):
        base[3][0], base[3][1] = make_group(base[3][0], base[3][1], c3[0], c3[1], i)
        for c4 in groups.values():
            base[4] = ["C", 1]
            for i in range(length):
                base[4][0], base[4][1] = make_group(
                    base[4][0], base[4][1], c4[0], c4[1], i
                )
                for c5 in groups.values():
                    base[5] = ["C", 1]
                    for i in range(length):
                        base[5][0], base[5][1] = make_group(
                            base[5][0], base[5][1], c5[0], c5[1], i
                        )
                        for c6 in groups.values():
                            base[6] = ["C", 1]
                            for i in range(length):
                                base[6][0], base[6][1] = make_group(
                                    base[6][0], base[6][1], c6[0], c6[1], i
                                )
                            print(
                                "OC(=O)C1="
                                + base[3][0]
                                + base[4][0]
                                + "="
                                + base[5][0]
                                + base[6][0]
                                + "=C1C(=O)O"
                            )
