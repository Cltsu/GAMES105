with open('./data/walk60.bvh', 'r' ) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith('Frame Time'):
            print(len(lines[i+1].split()))
