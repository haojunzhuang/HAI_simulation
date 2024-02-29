import numpy as np
import pandas as pd
import datetime
import re
from tqdm import tqdm


def _add_entry(df, id, date, from_department, to_department):
    df["id"].append(id)
    df["date"].append(pd.to_datetime(date, format="%Y-%m-%d"))
    df["from_department"].append(from_department)
    df["to_department"].append(to_department)

    return df


def read_movement_data(path_to_data, fill=False):
    """_summary_

    Parameters
    ----------
    path_to_data : _type_
        _description_
    fill : bool, optional
        If False, only include transition states. 
        If True, fill in the gaps with self-loops, by default False

    Returns
    -------
    _type_
        _description_
    """

    movement = pd.read_csv(path_to_data, on_bad_lines="warn", sep="|", engine="python")

    mvt = {"id": [""], "date": [""], "from_department": [""], "to_department": [""]}
    clock = pd.to_datetime("2100-01-01", format="%Y-%m-%d")
    department_to_exclude = [np.nan]
    last_row = {"deid_enc_id": "", "department_name": "", "end_time": "2100-01-01"}

    for _, row in tqdm(movement.iterrows()):
        if row["start_time"] is None:
            continue
        if row["department_name"] in department_to_exclude:
            continue

        current_start_t = pd.to_datetime(row["start_time"], format="%Y-%m-%d")
        current_end_t = pd.to_datetime(row["end_time"], format="%Y-%m-%d")

        if row["deid_enc_id"] != last_row["deid_enc_id"]:
            mvt = _add_entry(
                mvt,
                last_row["deid_enc_id"],
                last_row["end_time"],
                last_row["department_name"],
                "DISCHARGE",
            )
            mvt = _add_entry(
                mvt,
                row["deid_enc_id"],
                current_start_t,
                "ADMISSION",
                row["department_name"],
            )
        else:
            mvt = _add_entry(
                mvt,
                row["deid_enc_id"],
                current_start_t,
                last_row["department_name"],
                row["department_name"],
            )

        if fill:
            clock = current_start_t + datetime.timedelta(days=1)
            while clock <= current_end_t:
                mvt = _add_entry(
                    mvt,
                    row["deid_enc_id"],
                    clock,
                    row["department_name"],
                    row["department_name"],
                )
                clock += datetime.timedelta(days=1)

        last_row = row

    mvt = pd.DataFrame.from_dict(mvt)
    mvt = mvt.drop(0)

    return mvt


def compress_by_day(mvt):
    """Reduce the granularity of movement dataframe to a day.

    Parameters
    ----------
    mvt : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """


    mvt["date"] = pd.to_datetime(mvt["date"]).dt.date
    mvt_compressed = {
        "id": [""],
        "date": [""],
        "from_department": [""],
        "to_department": [""],
    }

    i = 1
    while i < len(mvt.index) - 1:
        now = mvt.loc[i]
        nxt = mvt.loc[i + 1]
        if now["id"] != nxt["id"] or now["date"] != nxt["date"]:
            mvt_compressed = _add_entry(
                mvt_compressed,
                now["id"],
                now["date"],
                now["from_department"],
                now["to_department"],
            )
            i += 1
            continue
        else:
            from_department = now["from_department"]
            while now["date"] == nxt["date"] and now["id"] == nxt["id"]:
                i += 1
                now = mvt.loc[i]
                nxt = mvt.loc[i + 1]
            mvt_compressed = _add_entry(
                mvt_compressed,
                now["id"],
                now["date"],
                from_department,
                now["to_department"],
            )
            i += 1

    mvt_compressed = pd.DataFrame.from_dict(mvt_compressed)
    mvt_compressed = mvt_compressed.drop(0)
    return mvt_compressed


def compress_self_loop(mvt):
    """Remove all self-loops of movement data.

    Parameters
    ----------
    mvt : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    return (
        mvt.drop(mvt[mvt["from_department"] == mvt["to_department"]].index)
        .reset_index(drop=True)
        .drop(0)
    )

def keep_departments_of_interest(df):
    vc = df.value_counts('to_department')
    blacklist = [w for w in vc.index if re.match(r'.*(MB|MZ|OAK)', w) or vc[w]<500]
    filtered_ids = df.groupby('id')['to_department'].apply(lambda x: any(item in blacklist for item in x)).reset_index()
    result = filtered_ids.loc[not filtered_ids['to_department'], 'id']

    # print('considering:', result)
    print('not considering:', blacklist)
    df = df[df['id'].isin(result)]
    df = df.reset_index(drop=True)

    return df
