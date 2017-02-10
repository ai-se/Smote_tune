import re
import sys

sys.dont_write_bytecode = True


class ReadFile(object):
    """
        1) This class contains everything related to parsing a file
        2) :params file_name --> absolute file path
           :params data --> Data object
    """
    def __init__(self, file_name):
        super(ReadFile, self).__init__()
        self.file_name = file_name
        self.raw_data = None
        if 'csv' == self.file_name.split(".")[-1]:
            self.raw_data = self.parse()
        else:
            print("Unsupported File: " + file_name)
            sys.exit()

    def parse(self):
        """
            Convert rows of strings to ints,floats, or strings
            as appropriate
        """

        def atoms(lst):
            return map(atom, lst)

        def atom(x):
            try:
                return int(x)
            except ValueError:
                try:
                    return float(x)
                except ValueError:
                    return x

        for row in self.rows(prep=atoms):
            yield row

    def rows(self, prep=None,
             whitespace='[\n\r\t]',
             comments='#.*',
             sep=","
             ):
        """
            Walk down comma separated values,
            skipping bad white space and blank lines
        """
        doomed = re.compile('(' + whitespace + '|' + comments + ')')
        with open(self.file_name) as fs:
            for line in fs:
                line = re.sub(doomed, "", line)
                if line:
                    row = map(lambda z: z.strip(), line.split(sep))
                    if len(row) > 0:
                        yield prep(row) if prep else row

    def build_table(self):
        """
            Removes all string columns and sets data object with filtered
            rows.
        """
        ignore_cols = []
        filtered_rows = []
        labels=[]
        for index, row in enumerate(self.raw_data):
            if index == 0:
                continue
            new_row = []
            for col, val in enumerate(row):
                if col in ignore_cols:
                    continue
                try:
                    val = float(val)
                    new_row.append(val)
                except ValueError:
                    ignore_cols.append(col)
            filtered_rows.append(new_row[:-1])
            labels.append(new_row[-1])
        label = [1 if l > 0 else 0 for l in labels]
        return filtered_rows,label
