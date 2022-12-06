anova = ['service', 'flag', 'logged_in', 'count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'same_srv_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'difficulty_level', 'attack_class']
info_gain = ['src_bytes', 'dst_bytes', 'service', 'flag', 'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'logged_in', 'dst_host_serror_rate', 'count', 'serror_rate', 'dst_host_srv_serror_rate', 'srv_serror_rate']
ri1 = ['flag', 'difficulty_level', 'diff_srv_rate', 'same_srv_rate', 'dst_host_same_srv_rate', 'count', 'dst_host_srv_count', 'protocol_type', 'logged_in', 'service', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_count']
ri2 = ['difficulty_level', 'flag', 'same_srv_rate', 'dst_host_srv_count', 'diff_srv_rate', 'dst_host_diff_srv_rate', 'count', 'logged_in', 'dst_host_same_srv_rate', 'protocol_type', 'dst_host_same_src_port_rate', 'service', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'srv_serror_rate']
ri3 = ['difficulty_level', 'flag', 'same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'count', 'diff_srv_rate', 'dst_host_srv_count', 'logged_in', 'protocol_type', 'service', 'serror_rate', 'dst_host_same_src_port_rate', 'dst_host_count', 'dst_host_rerror_rate']
rfe = ['protocol_type', 'service', 'flag', 'logged_in', 'count', 'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'difficulty_level']

rf = [ri1, ri2, ri3]
allfs = [anova, info_gain, ri1, ri2, ri3, rfe]

common = set(anova)
common_i = set(anova)

for i in allfs:
    common = common | set(i)
    common_i = common_i & set(i)

print(list(common))
print()
print(list(common_i))
common_rf = set(ri1)
for i in rf:
    common_rf = common_rf | set(i)
#print(common_rf)

""" common features in all of the above
['same_srv_rate', 'flag', 'dst_host_same_srv_rate', 'count', 'dst_host_srv_count', 'service', 'logged_in']
"""

"""
'protocol_type'
'service'
'flag'
'dst_bytes'
'count'
'srv_count'
'serror_rate'
'same_srv_rate'
'dst_host_count'
'dst_host_same_srv_rate'
'dst_host_diff_srv_rate'
'dst_host_srv_diff_host_rate'
'dst_host_serror_rate'
'dst_host_srv_serror_rate'
'difficulty_level'
'dst_host_rerror_rate'
'dst_host_srv_count'
'wrong_fragment'
'duration'
'dst_host_same_src_port_rate'
'logged_in'
'src_bytes'
'diff_srv_rate'
'is_guest_login'
'srv_serror_rate'
'dst_host_srv_rerror_rate'
'srv_diff_host_rate'
'hot'
'num_root'
'rerror_rate'
'srv_rerror_rate'
'num_compromised'
'num_shells'
'num_failed_logins'
'num_file_creations'
'su_attempted'
'land'
'root_shell'
'num_access_files'
'num_outbound_cmds'
'urgent'
'is_host_login'
"""