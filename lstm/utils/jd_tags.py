new_tag_list = ['B-prov', 'B-city', 'B-district', 'B-devzone', 'B-town',
                'B-community', 'B-village_group', 'B-roadno', 'B-road',
                'B-poi', 'B-subpoi', 'B-houseno', 'B-cellno', 'B-floorno',
                'B-roomno', 'B-detail', 'B-assist', 'B-distance',
                'B-intersection', 'B-redundant',
                'I-prov', 'I-city', 'I-district', 'I-devzone', 'I-town',
                'I-community', 'I-village_group', 'I-roadno', 'I-road', 'I-poi',
                'I-subpoi', 'I-houseno', 'I-cellno', 'I-floorno', 'I-roomno',
                'I-detail', 'I-assist', 'I-distance', 'I-intersection',
                'I-redundant',
                'E-prov', 'E-city', 'E-district', 'E-devzone', 'E-town',
                'E-community', 'E-village_group', 'E-roadno', 'E-road',
                'E-poi', 'E-subpoi', 'E-houseno', 'E-cellno', 'E-floorno',
                'E-roomno', 'E-detail', 'E-assist', 'E-distance', 'E-intersection',
                'E-redundant',
                'S-assist', 'S-poi', 'S-intersection', 'S-district', 'S-community',
                'O']

tag2id = {
  "B-prov": 0,
  "B-city": 1,
  "B-district": 2,
  "B-devzone": 3,
  "B-town": 4,
  "B-community": 5,
  "B-village_group": 6,
  "B-roadno": 7,
  "B-road": 8,
  "B-poi": 9,
  "B-subpoi": 10,
  "B-houseno": 11,
  "B-cellno": 12,
  "B-floorno": 13,
  "B-roomno": 14,
  "B-detail": 15,
  "B-assist": 16,
  "B-distance": 17,
  "B-intersection": 18,
  "B-redundant": 19,
  "I-prov": 20,
  "I-city": 21,
  "I-district": 22,
  "I-devzone": 23,
  "I-town": 24,
  "I-community": 25,
  "I-village_group": 26,
  "I-roadno": 27,
  "I-road": 28,
  "I-poi": 29,
  "I-subpoi": 30,
  "I-houseno": 31,
  "I-cellno": 32,
  "I-floorno": 33,
  "I-roomno": 34,
  "I-detail": 35,
  "I-assist": 36,
  "I-distance": 37,
  "I-intersection": 38,
  "I-redundant": 39,
  "E-prov": 40,
  "E-city": 41,
  "E-district": 42,
  "E-devzone": 43,
  "E-town": 44,
  "E-community": 45,
  "E-village_group": 46,
  "E-roadno": 47,
  "E-road": 48,
  "E-poi": 49,
  "E-subpoi": 50,
  "E-houseno": 51,
  "E-cellno": 52,
  "E-floorno": 53,
  "E-roomno": 54,
  "E-detail": 55,
  "E-assist": 56,
  "E-distance": 57,
  "E-intersection": 58,
  "E-redundant": 59,
  "S-assist": 60,
  "S-poi": 61,
  "S-intersection": 62,
  "S-district": 63,
  "S-community": 64,
  "O": 65
}

id2tag = {
  "0": "B-prov",
  "1": "B-city",
  "2": "B-district",
  "3": "B-devzone",
  "4": "B-town",
  "5": "B-community",
  "6": "B-village_group",
  "7": "B-roadno",
  "8": "B-road",
  "9": "B-poi",
  "10": "B-subpoi",
  "11": "B-houseno",
  "12": "B-cellno",
  "13": "B-floorno",
  "14": "B-roomno",
  "15": "B-detail",
  "16": "B-assist",
  "17": "B-distance",
  "18": "B-intersection",
  "19": "B-redundant",
  "20": "I-prov",
  "21": "I-city",
  "22": "I-district",
  "23": "I-devzone",
  "24": "I-town",
  "25": "I-community",
  "26": "I-village_group",
  "27": "I-roadno",
  "28": "I-road",
  "29": "I-poi",
  "30": "I-subpoi",
  "31": "I-houseno",
  "32": "I-cellno",
  "33": "I-floorno",
  "34": "I-roomno",
  "35": "I-detail",
  "36": "I-assist",
  "37": "I-distance",
  "38": "I-intersection",
  "39": "I-redundant",
  "40": "E-prov",
  "41": "E-city",
  "42": "E-district",
  "43": "E-devzone",
  "44": "E-town",
  "45": "E-community",
  "46": "E-village_group",
  "47": "E-roadno",
  "48": "E-road",
  "49": "E-poi",
  "50": "E-subpoi",
  "51": "E-houseno",
  "52": "E-cellno",
  "53": "E-floorno",
  "54": "E-roomno",
  "55": "E-detail",
  "56": "E-assist",
  "57": "E-distance",
  "58": "E-intersection",
  "59": "E-redundant",
  "60": "S-assist",
  "61": "S-poi",
  "62": "S-intersection",
  "63": "S-district",
  "64": "S-community",
  "65": "O"
}


des_list = ['????????????', '??????', '????????????', '????????????', 'Email', '??????', '?????????', '????????????', '????????????', '????????????', '????????????', '??????????????????', '????????????',
            '??????????????????', '????????????', '????????????', '??????????????????', '???????????????', '??????????????????', '????????????', '????????????', '????????????', '????????????', '??????????????????', '????????????',
            '??????????????????', '??????', '????????????', '??????????????????']

tag_list = ['O', 'B-MAJR', 'I-MAJR', 'B-TELL', 'I-TELL', 'B-SCHL', 'I-SCHL', 'B-AOPE', 'I-AOPE', 'B-EMAL', 'I-EMAL',
            'B-WECH', 'I-WECH', 'B-COMN', 'I-COMN', 'B-DEGR', 'I-DEGR', 'B-ADDR', 'I-ADDR', 'B-COMI', 'I-COMI',
            'B-RESP', 'I-RESP', 'B-PSKL', 'I-PSKL', 'B-SALR', 'I-SALR', 'B-REQM', 'I-REQM', 'B-YOWK', 'I-YOWK',
            'B-DEPN', 'I-DEPN', 'B-JEXP', 'I-JEXP', 'B-AEXP', 'I-AEXP', 'B-INDU', 'I-INDU', 'B-LANG', 'I-LANG',
            'B-TITL', 'I-TITL', 'B-RWAD', 'I-RWAD', 'B-MSKL', 'I-MSKL', 'B-AGEE', 'I-AGEE', 'B-ASKL', 'I-ASKL',
            'B-CERT', 'I-CERT', 'B-GEND', 'I-GEND', 'B-SSKL', 'I-SSKL', 'B-JOBR', 'I-JOBR', 'B-JOBT', 'I-JOBT',
            'B-WOKT', 'I-WOKT', 'B-REMK', 'I-REMK', 'B-OTRE', 'I-OTRE', 'B-AOTR', 'I-AOTR', 'B-JRTI', 'I-JRTI',
            'B-JDTI', 'I-JDTI']

des2tag = {
    "????????????": "MAJR",
    "??????": "TELL",
    "????????????": "SCHL",
    "????????????": "AOPE",
    "Email": "EMAL",
    "??????": "WECH",
    "?????????": "COMN",
    "????????????": "DEGR",
    "????????????": "ADDR",
    "????????????": "COMI",
    "????????????": "RESP",
    "??????????????????": "PSKL",
    "????????????": "SALR",
    "??????????????????": "REQM",
    "????????????": "YOWK",
    "????????????": "DEPN",
    "??????????????????": "JEXP",
    "??????????????????": "AEXP",
    "????????????": "INDU",
    "????????????": "LANG",
    "?????????": "TITL",
    "????????????": "RWAD",
    "??????????????????": "MSKL",
    "????????????": "AGEE",
    "??????????????????": "ASKL",
    "??????": "CERT",
    "????????????": "GEND",
    "??????????????????": "SSKL",
    "????????????": "JOBR",
    "????????????": "JOBT",
    "??????????????????": "WOKT",
    "??????": "REMK",
    "????????????": "OTRE",
    "???????????????": "AOTR",
    "??????????????????": "JRTI",
    "??????????????????": "JDTI"
}

tag2des = { 'MAJR': '????????????',
            'TELL': '??????',
            'SCHL': '????????????',
            'AOPE': '????????????',
            'EMAL': 'Email',
            'WECH': '??????',
            'COMN': '?????????',
            'DEGR': '????????????',
            'ADDR': '????????????',
            'COMI': '????????????',
            'RESP': '????????????',
            'PSKL': '??????????????????',
            'SALR': '????????????',
            'REQM': '??????????????????',
            'YOWK': '????????????',
            'DEPN': '????????????',
            'JEXP': '??????????????????',
            'AEXP': '??????????????????',
            'INDU': '????????????',
            'LANG': '????????????',
            'TITL': '????????????',
            'RWAD': '????????????',
            'MSKL': '??????????????????',
            'AGEE': '????????????',
            'ASKL': '??????????????????',
            'CERT': '??????',
            'GEND': '????????????',
            'SSKL': '??????????????????',
            "JOBT": '????????????',
            "JOBR": '????????????',
            "WOKT": '??????????????????',
            "REMK": '??????',
            "OTRE": '????????????',
            "AOTR": '???????????????',
            "JRTI": '??????????????????',
            "JDTI": '??????????????????',
}

