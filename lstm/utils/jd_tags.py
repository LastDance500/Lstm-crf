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


des_list = ['专业要求', '电话', '学校要求', '招聘人数', 'Email', '微信', '公司名', '学历要求', '工作地址', '公司介绍', '职位职能', '专业技能要求', '薪资水平',
            '任职公司要求', '工作年限', '部门名称', '职能经验要求', '技能关键词', '行业经验要求', '公司行业', '语言要求', '职位名称', '福利待遇', '管理能力要求', '年龄要求',
            '加分技能要求', '证书', '性别要求', '软性技能要求']

tag_list = ['O', 'B-MAJR', 'I-MAJR', 'B-TELL', 'I-TELL', 'B-SCHL', 'I-SCHL', 'B-AOPE', 'I-AOPE', 'B-EMAL', 'I-EMAL',
            'B-WECH', 'I-WECH', 'B-COMN', 'I-COMN', 'B-DEGR', 'I-DEGR', 'B-ADDR', 'I-ADDR', 'B-COMI', 'I-COMI',
            'B-RESP', 'I-RESP', 'B-PSKL', 'I-PSKL', 'B-SALR', 'I-SALR', 'B-REQM', 'I-REQM', 'B-YOWK', 'I-YOWK',
            'B-DEPN', 'I-DEPN', 'B-JEXP', 'I-JEXP', 'B-AEXP', 'I-AEXP', 'B-INDU', 'I-INDU', 'B-LANG', 'I-LANG',
            'B-TITL', 'I-TITL', 'B-RWAD', 'I-RWAD', 'B-MSKL', 'I-MSKL', 'B-AGEE', 'I-AGEE', 'B-ASKL', 'I-ASKL',
            'B-CERT', 'I-CERT', 'B-GEND', 'I-GEND', 'B-SSKL', 'I-SSKL', 'B-JOBR', 'I-JOBR', 'B-JOBT', 'I-JOBT',
            'B-WOKT', 'I-WOKT', 'B-REMK', 'I-REMK', 'B-OTRE', 'I-OTRE', 'B-AOTR', 'I-AOTR', 'B-JRTI', 'I-JRTI',
            'B-JDTI', 'I-JDTI']

des2tag = {
    "专业要求": "MAJR",
    "电话": "TELL",
    "学校要求": "SCHL",
    "招聘人数": "AOPE",
    "Email": "EMAL",
    "微信": "WECH",
    "公司名": "COMN",
    "学历要求": "DEGR",
    "工作地址": "ADDR",
    "公司介绍": "COMI",
    "职位职能": "RESP",
    "专业技能要求": "PSKL",
    "薪资水平": "SALR",
    "任职公司要求": "REQM",
    "工作年限": "YOWK",
    "部门名称": "DEPN",
    "职能经验要求": "JEXP",
    "行业经验要求": "AEXP",
    "公司行业": "INDU",
    "语言要求": "LANG",
    "职位名": "TITL",
    "福利待遇": "RWAD",
    "管理能力要求": "MSKL",
    "年龄要求": "AGEE",
    "加分技能要求": "ASKL",
    "证书": "CERT",
    "性别要求": "GEND",
    "软性技能要求": "SSKL",
    "岗位职责": "JOBR",
    "职位类型": "JOBT",
    "工作时长要求": "WOKT",
    "备注": "REMK",
    "其他要求": "OTRE",
    "其他加分项": "AOTR",
    "岗位职责标题": "JRTI",
    "岗位要求标题": "JDTI"
}

tag2des = { 'MAJR': '专业要求',
            'TELL': '电话',
            'SCHL': '学校要求',
            'AOPE': '招聘人数',
            'EMAL': 'Email',
            'WECH': '微信',
            'COMN': '公司名',
            'DEGR': '学历要求',
            'ADDR': '工作地址',
            'COMI': '公司介绍',
            'RESP': '职位职能',
            'PSKL': '专业技能要求',
            'SALR': '薪资水平',
            'REQM': '任职公司要求',
            'YOWK': '工作年限',
            'DEPN': '部门名称',
            'JEXP': '职能经验要求',
            'AEXP': '行业经验要求',
            'INDU': '公司行业',
            'LANG': '语言要求',
            'TITL': '职位名称',
            'RWAD': '福利待遇',
            'MSKL': '管理能力要求',
            'AGEE': '年龄要求',
            'ASKL': '加分技能要求',
            'CERT': '证书',
            'GEND': '性别要求',
            'SSKL': '软性技能要求',
            "JOBT": '岗位职责',
            "JOBR": '职位类型',
            "WOKT": '工作时长要求',
            "REMK": '备注',
            "OTRE": '其他要求',
            "AOTR": '其他加分项',
            "JRTI": '岗位职责标题',
            "JDTI": '岗位要求标题',
}
