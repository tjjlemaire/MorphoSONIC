import os
import re


def createSubModFile(fpath, suffix):
    # Read lines from MOD file
    with open(fpath, 'r') as fh:
        lines = [line.rstrip() for line in fh]

    # Define search patterns
    blocks = ['NEURON', 'PARAMETER', 'BREAKPOINT']
    block_entries = {name: re.compile(name + '[\s]*{.*') for name in blocks}
    block_exit = re.compile('}(.*)')
    is_in_block = {name: False for name in blocks}
    SUFFIX = re.compile('([\s]*SUFFIX) ([A-Za-z]*)(.*)')
    CURRENT_def = re.compile('[\s]*NONSPECIFIC_CURRENT ([A-Za-z]*).*')

    # Loop through lines
    newlines = []
    currents = []
    for i, line in enumerate(lines):
        # Check for blocks entry/exit
        for block in blocks:
            if block_entries[block].fullmatch(line):
                print('line {}: {} block entry'.format(i + 1, block))
                is_in_block[block] = True
            if is_in_block[block] and block_exit.fullmatch(line):
                # Add weight to RANGE variables definition in NEURON block
                if block == 'NEURON':
                    print('Declaring "w" RANGE variable in NEURON block')
                    newlines.append('')
                    newlines.append('    RANGE w')
                # Define weight in PARAMETER block
                if block == 'PARAMETER':
                    print('Defining "w" RANGE variable in PARAMETER block')
                    newlines.append('')
                    newlines.append('    w : relative weight of mechanism in section')
                print('line {}: {} block exit'.format(i + 1, block))
                is_in_block[block] = False

        if is_in_block['NEURON']:
            # Modify SUFFIX line
            match = SUFFIX.fullmatch(line)
            if match:
                print('line {}: Modifying NEURON SUFFIX'.format(i))
                line = '{} {}_{}{}'.format(match.group(1), match.group(2), suffix, match.group(3))

            # Identify membrane currents
            match = CURRENT_def.fullmatch(line)
            if match:
                print('line {}: {} current definition'.format(i, match.group(1)))
                currents.append(match.group(1))

        # Add "w" factor to membrane currents assignments
        if is_in_block['BREAKPOINT']:
            for current in currents:
                CURRENT_assgnmt = re.compile('([\s]*' + current + '[\s]*=[\s]*)([^:]*)(.*)')
                match = CURRENT_assgnmt.fullmatch(line)
                if match:
                    print('line {}: adding "w" factor to {} current assignment'.format(i, current))
                    line = match.group(1) + 'w * (' + match.group(2) + ')' + match.group(3)

        # Populate new lines list
        newlines.append(line)

    # Write modified lines into new MOD file
    suffix_fpath = '{}_{}.mod'.format(os.path.splitext(fpath)[0], suffix)
    with open(suffix_fpath, "w") as fh:
        fh.write('\n'.join(newlines) + '\n')


if __name__ == '__main__':
    root = 'C:/Users/Theo/Documents/ExSONIC/ExSONIC/nmodl'
    neurons = ['RS', 'FS', 'LTS', 'RE', 'TC']
    suffixes = ['AM', 'S']
    for neuron in neurons:
        src = os.path.join(root, '{}.mod'.format(neuron))
        for suffix in suffixes:
            createSubModFile(src, suffix)
