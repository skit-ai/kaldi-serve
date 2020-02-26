"""
Script for parsing server debug logs.

Usage:
  parse_logs.py <logs-file> <metrics-csv>
"""

from docopt import docopt

import pandas as pd


def main(args):
    df = parse_logs(args["<logs-file>"])
    df.to_csv(args["<metrics-csv>"], index=False)


def parse_logs(logs_file):
    with open(logs_file, "r", encoding="utf-8") as f:
        logs = f.read().split("\n")
    
    uuids = []

    for log in logs:
        if "decoder acquired" in log:
            parts = log.split()
            uuids.append(parts[parts.index("uuid:") + 1])

    uuids = list(set(uuids))

    df = pd.DataFrame(columns=["uuid", "chunk#", "read", "accepted", "decoded", "computed"])

    for uuid in uuids:
        l_logs = list(filter(lambda x: uuid in x, logs))
        chunks = []
        for log in l_logs:
            if "chunk" in log and "received" in log:
                parts = log.split()
                chunks.append("chunk " + parts[parts.index("chunk") + 1])
        chunks = sorted(list(set(chunks)))

        for chunk in chunks:
            chunk_logs = list(filter(lambda x: chunk in x, l_logs))
            idx = l_logs.index(chunk_logs[0])

            read = float(l_logs[idx + 1].split()[-1].replace("ms", ""))
            accepted = float(l_logs[idx + 2].split()[-1].replace("ms", ""))
            decoded = float(l_logs[idx + 3].split()[-1].replace("ms", ""))
            computed = float(chunk_logs[1].split()[-1].replace("ms", ""))

            df = df.append({
                "uuid": uuid,
                "chunk#": chunk,
                "read": read,
                "accepted": accepted,
                "decoded": decoded,
                "computed": computed
            }, ignore_index=True)

    return df


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)