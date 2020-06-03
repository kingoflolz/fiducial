import imageio
import numpy as np

def generate(size, name, data):
    assert size > 1
    side_length = (3 + 3 * size) * 2

    tag = np.ones((side_length, side_length, 3), dtype=np.uint8)
    tag.fill(255)

    data_blocks = []

    for i in range(size):
        for j in range(size):
            data_blocks.append((j, i))

    data_blocks.remove((0, 0))
    data_blocks.remove((size-1, 0))

    # add border
    for i in range(side_length):
        tag[i, 0, :] = 0
        tag[i, 1, :] = 0

        tag[i, -1, :] = 0
        tag[i, -2, :] = 0

        tag[0, i, :] = 0
        tag[1, i, :] = 0

        tag[-1, i, :] = 0
        tag[-2, i, :] = 0

    tag[4:8, 4:8, :] = 0
    tag[4:8, -8:-4, :] = 0

    for pos in data_blocks:
        block_data = data & 0b11
        data >>= 2

        sub_pos = [block_data & 1, (block_data & 2) >> 1]

        x = 4 + pos[1] * 6 + sub_pos[0] * 1
        y = 4 + pos[0] * 6 + sub_pos[1] * 1

        tag[x:x+3, y:y+3, :] = 0
        # tag[x, y, :] = 127
        # tag[x + 2, y, :] = 127
        # tag[x, y + 2, :] = 127
        # tag[x + 2, y + 2, :] = 127
    imageio.imwrite(name, tag)

def generate_class(size, upto = 5000):
    data_blocks = size ** 2 - 2
    total_tags = 4 ** data_blocks
    print(f"generating tag with size {size}, {data_blocks * 4} bits and {total_tags} tags")
    for i in range(min(upto, total_tags)):
        generate(size, f"generated/tag{size}/tag{size}_{i:05}.png", i)


# generate_class(2)
# generate_class(3)
generate_class(4)

i = 2**28 - 1
generate(4, f"generated/tag4/tag4_{i:05}.png", i)
