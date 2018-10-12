# -*- coding: utf-8 -*-
import json
import os
from config import label_names

if __name__ == '__main__':
    with open('README.template', 'r', encoding="utf-8") as file:
        template = file.readlines()
    template = ''.join(template)

    filename = 'result.json'
    if os.path.isfile(filename):
        with open(filename, 'r', encoding="utf-8") as file:
            result = json.load(file)

    for i in range(10):
        template = template.replace('$(content_{})'.format(i), result[i]['content'])
        for j, label_name in enumerate(label_names):
            template = template.replace('$(label_{}_{})'.format(i, j), str(result[i]['labels'][j]))

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
