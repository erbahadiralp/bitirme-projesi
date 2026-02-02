import pandas as pd
import os
import ast

def load_and_prepare_data(base_path, sampling_rate=100):
    """
    PTB-XL verilerini yükler, meta verileri işler ve süper sınıfları ekler.
    """
    db_path = os.path.join(base_path, 'ptbxl_database.csv')
    df = pd.read_csv(db_path, index_col='ecg_id')

    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df['filename'] = df.filename_hr if sampling_rate == 500 else df.filename_lr
    df['filepath'] = df.filename.apply(lambda x: os.path.join(base_path, x))

    scp_path = os.path.join(base_path, 'scp_statements.csv')
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
    
    df_filtered = df[df.superdiagnostic.apply(lambda x: len(x) > 0)].copy()
    
    if df_filtered.empty:
        return None

    df_filtered['main_diagnostic'] = df_filtered.superdiagnostic.apply(lambda x: x[0])

    target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    df_final = df_filtered[df_filtered.main_diagnostic.isin(target_classes)].copy()
    
    return df_final