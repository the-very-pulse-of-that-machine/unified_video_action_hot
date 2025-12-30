# -*- coding: utf-8 -*-

# 各模型的耗时数据（毫秒）
models = {
    "origin": {
        "model.encoder_blocks": 72078.13774736319,
        "model.decoder_blocks": 71507.9990308471,
        "model.cross_attention": 0,
        "model.decoder_embed": 137.65558338165283,
        "model.decoder_norm": 10.811391949653625,
        "model.diffactloss": 80537.89702422172,
        "model.encoder_norm": 11.843392014503479,
        "model.proj_cond_x_layer": 422.80358505249023,
        "model.z_proj_cond": 16.22732800245285,
        "model.z_proj_ln": 11.274047940969467,
        "vae_model.encoder": 71383.30335958372,
        "vae_model.quant_conv": 5.974976003170013
    },
    "hot": {
        "model.encoder_blocks": 37369.2895211624,
        "model.decoder_blocks": 71437.07150291558,
        "model.cross_attention": 1044.7117742542177,
        "model.decoder_embed": 137.33091115951538,
        "model.decoder_norm": 10.794751971960068,
        "model.diffactloss": 80125.48282362893,
        "model.encoder_norm": 10.038592010736465,
        "model.proj_cond_x_layer": 422.6297941207886,
        "model.z_proj_cond": 16.14675199985504,
        "model.z_proj_ln": 11.220032006502151,
        "vae_model.encoder": 71349.17911068164,
        "vae_model.quant_conv": 6.067488074302673
    },
    "expand_hot": {
        "model.encoder_blocks": 37517.48002218362,
        "model.decoder_blocks": 20349.77010058798,
        "model.cross_attention": 1049.5251508790534,
        "model.decoder_embed": 43.38035190105438,
        "model.decoder_norm": 10.03865596652031,
        "model.diffactloss": 82292.43725343421,
        "model.encoder_norm": 4.178847998380661,
        "model.proj_cond_x_layer": 417.3514232635498,
        "model.z_proj_cond": 16.796831995248795,
        "model.z_proj_ln": 11.414464145898819,
        "vae_model.encoder": 70016.72424228303,
        "vae_model.quant_conv": 6.9401280879974365
    },
    "expand_hot_v2_ratio01": {
        "model.encoder_blocks": 29438.131226338446,
        "model.decoder_blocks": 7467.756444172002,
        "model.cross_attention": 760.7216304654721,
        "model.decoder_embed": 16.294624090194702,
        "model.decoder_norm": 10.064703941345215,
        "model.diffactloss": 82700.81052655913,
        "model.encoder_norm": 2.0123200081288815,
        "model.proj_cond_x_layer": 421.56220531463623,
        "model.z_proj_cond": 34.73785516619682,
        "model.z_proj_ln": 18.208415925502777,
        "vae_model.encoder": 74289.1608919932,
        "vae_model.quant_conv": 7.168927900493145
    },
    "expand_hot_v2_ratio005": {
        "model.encoder_blocks": 27648.596304381266,
        "model.decoder_blocks": 4494.070657134056,
        "model.cross_attention": 706.7092172957491,
        "model.decoder_embed": 10.013312011957169,
        "model.decoder_norm": 10.058527946472168,
        "model.diffactloss": 80924.44791250676,
        "model.encoder_norm": 1.5117120034992695,
        "model.proj_cond_x_layer": 425.9417600631714,
        "model.z_proj_cond": 39.18025657534599,
        "model.z_proj_ln": 17.567647755146027,
        "vae_model.encoder": 73854.1869065885,
        "vae_model.quant_conv": 7.698111817240715
    },
    "e_h_ttt": {
        "model.encoder_blocks": 38993.62355705071,
        "model.decoder_blocks": 20183.212069073692,
        "model.cross_attention": 966.6425268969033,
        "model.decoder_embed": 37.79465591907501,
        "model.decoder_norm": 10.063007980585098,
        "model.diffactloss": 81592.76851461455,
        "model.encoder_norm": 3.698303982615471,
        "model.proj_cond_x_layer": 424.5146255493164,
        "model.z_proj_cond": 270.2717131078243,
        "model.z_proj_ln": 12.738624036312103,
        "vae_model.encoder": 71042.34284531372,
        "vae_model.quant_conv": 6.302240125834942
    }
}

def calculate_speedup(origin, target):
    # 总耗时
    total_origin = sum(origin.values())
    total_target = sum(target.values())
    total_speedup = total_origin / total_target

    # Transformer耗时（encoder_blocks + decoder_blocks）
    trans_origin = origin["model.encoder_blocks"] + origin["model.decoder_blocks"]
    trans_target = target["model.encoder_blocks"] + target["model.decoder_blocks"]
    trans_speedup = trans_origin / trans_target

    return total_speedup, trans_speedup

# 输出提速比例
print("模型提速比例（相对于origin）：")
for name, data in models.items():
    if name == "origin":
        continue
    total_speedup, trans_speedup = calculate_speedup(models["origin"], data)
    print(f"{name}: 总耗时提速 {total_speedup:.2f}x, Transformer 提速 {trans_speedup:.2f}x")
