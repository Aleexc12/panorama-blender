import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import argparse

def cylindricalWarpImage(img, f):
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    cx = w / 2.0
    cy = h / 2.0

    cylinder = np.full((h, w, 3), -1.0, dtype=np.float32)

    for x_cyl in range(w):
        theta = (x_cyl - cx) / f
        X = np.tan(theta) * f

        for y_cyl in range(h):
            y_ = (y_cyl - cy)
            denom = np.sqrt(X*X + f*f)
            Y = y_ * denom / f

            orig_x = int(Y + cy)
            orig_y = int(X + cx)

            if 0 <= orig_x < h and 0 <= orig_y < w:
                cylinder[y_cyl, x_cyl] = img_f[orig_x, orig_y]
    return cylinder

def cropCylinder(cyl_img):
    h, w, _ = cyl_img.shape
    mask_valid = (cyl_img[:, :, 0] != -1) | \
                 (cyl_img[:, :, 1] != -1) | \
                 (cyl_img[:, :, 2] != -1)
    coords = np.argwhere(mask_valid)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return cyl_img[y_min:y_max+1, x_min:x_max+1].copy()

def convertToUint8(fimg):
    out = np.clip(fimg, 0, 255).astype(np.uint8)
    return out

def warpAndCrop(img, focal):
    cyl = cylindricalWarpImage(img, focal)
    crp = cropCylinder(cyl)
    if crp is None:
        return None
    return convertToUint8(crp)

def siftFlannTranslation(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kps1, desc1 = sift.detectAndCompute(gray1, None)
    kps2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kps1) < 2 or len(kps2) < 2:
        return 0.0, 0.0

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_all = flann.knnMatch(desc1, desc2, k=2)
    good = []
    ratio_thresh = 0.7
    for m, n in matches_all:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0.0, 0.0

    pts1 = np.float32([(kps1[m.queryIdx].pt[1], kps1[m.queryIdx].pt[0]) for m in good])
    pts2 = np.float32([(kps2[m.trainIdx].pt[1], kps2[m.trainIdx].pt[0]) for m in good])

    n_iters = 300
    best_inliers = -1
    best_dx, best_dy = 0.0, 0.0
    rng = np.random.default_rng(42)

    for _ in range(n_iters):
        idx_rand = rng.integers(len(good))
        x1, y1 = pts1[idx_rand]
        x2, y2 = pts2[idx_rand]
        tdx = x2 - x1
        tdy = y2 - y1

        dist_thresh = 5.0
        dxs = pts2[:, 0] - pts1[:, 0]
        dys = pts2[:, 1] - pts1[:, 1]
        dist_sq = (dxs - tdx)**2 + (dys - tdy)**2
        inliers_count = np.count_nonzero(dist_sq < dist_thresh*dist_thresh)

        if inliers_count > best_inliers:
            best_inliers = inliers_count
            best_dx = tdx
            best_dy = tdy
    best_dx *= -1
    best_dy *= -1

    return best_dx, best_dy


def stitch_preprocess(inputDir, outputDir, focal_length=1700.0):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    imgListPath = os.path.join(inputDir, "img_list.txt")
    with open(imgListPath, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    if len(lines) < 2:
        print("[❌ ERROR] 'img_list.txt' must contain at least 2 lines (focal length and one image filename).")
        return
    try:
        focal = float(lines[0])
        print(f"[📷 INFO] Using focal length from file: {focal}")
    except:
        focal = focal_length
        print(f"[📷 INFO] Using fallback focal length: {focal}")
    imgNames = lines[1:]
    pics = []
    for i, imgName in enumerate(imgNames):
        path = os.path.join(inputDir, imgName)
        img = cv2.imread(path)
        if img is None:
            print(f"[❌ ERROR] Could not read {path}")
            return
        warped = warpAndCrop(img, focal)
        if warped is None:
            print(f"[❌ ERROR] Projection failed for {imgName}")
            return
        pics.append(warped)
    n = len(pics)
    dx = [0.0] * (n - 1)
    dy = [0.0] * (n - 1)
    for i in range(n - 1):
        dx[i], dy[i] = siftFlannTranslation(pics[i], pics[i + 1])
    cdx = [0.0] * n
    cdy = [0.0] * n
    for i in range(1, n):
        cdx[i] = cdx[i - 1] + dx[i - 1]
        cdy[i] = cdy[i - 1] + dy[i - 1]
    metadataPath = os.path.join(outputDir, "metadata.txt")
    with open(metadataPath, "w") as f:
        for i in range(n):
            outName = f"fin_img{i}.jpg"
            outFull = os.path.join(outputDir, outName)
            cv2.imwrite(outFull, pics[i])
            shiftY = int(round(cdy[i]))
            outLine = f"{outFull} {shiftY}\n"
            f.write(outLine)
    print(f"[✅ INFO] Metadata saved to {metadataPath}")
    print("[✅ INFO] Preprocessing completed. Ready for blending.")


def equalize_brightness(pics, cum_yshift):

    imgcnt = len(pics)
    for i in range(1, imgcnt):
        overlap = cum_yshift[i-1] + pics[i-1].shape[1] - cum_yshift[i]
        if overlap <= 0:
            continue
        rows_ = pics[i].shape[0]
        inten_del = 0.0
        cnt = 0.0
        for x in range(rows_):
            if x < 2 or x >= (rows_ - 2):
                for y in range(int(overlap)):
                    yy = y + (cum_yshift[i] - cum_yshift[i-1])
                    if y<0 or y>=pics[i].shape[1]:
                        continue
                    if yy<0 or yy>=pics[i-1].shape[1]:
                        continue
                    my_clr = pics[i][x,y]
                    fin_clr= pics[i-1][x,yy]
                    my_inten  = 0.114*my_clr[0] + 0.587*my_clr[1] + 0.299*my_clr[2]
                    fin_inten = 0.114*fin_clr[0] + 0.587*fin_clr[1] + 0.299*fin_clr[2]
                    inten_del += (my_inten - fin_inten)
                    cnt +=1.0
        if cnt>0:
            inten_del /= cnt
            pics[i] -= inten_del
            np.clip(pics[i], 0,255, out=pics[i])

def compute_graph_cut(diff, r_i, overlap_w):

    import heapq
    choice = np.zeros(diff.shape, dtype=np.int32)
    Q = []
    lastcol = overlap_w - 1

    # Insertar bordes
    for row_ in range(r_i):
        for off_ in [0,1,2,3,4]:
            if (0+off_) < overlap_w:
                Q.append((-1000, (-1, row_*overlap_w + (0+off_))))
            if (lastcol-off_) >= 0:
                Q.append((-1000, (1, row_*overlap_w + (lastcol-off_))))

    heapq.heapify(Q)

    dx4 = [0, 0, 1, -1]
    dy4 = [1, -1, 0, 0]

    while Q:
        neg_cost, (clr, idx_) = heapq.heappop(Q)
        xx = idx_ // overlap_w
        yy = idx_ % overlap_w
        if choice[xx,yy] != 0:
            continue
        choice[xx,yy] = clr

        for k_ in range(4):
            nx = xx + dx4[k_]
            ny = yy + dy4[k_]
            if nx<0 or nx>=r_i or ny<0 or ny>=overlap_w:
                continue
            if choice[nx, ny] == 0:
                cost_ = diff[nx, ny]
                heapq.heappush(Q, (-cost_, (clr, nx*overlap_w+ny)))

    return choice

def stitchImageGraphCutOnly(i, pics, canvas, y_shift):

    r_i, c_i = pics[i].shape[:2]
    offset = y_shift[i]

    overlap_w = int(y_shift[i-1] + pics[i-1].shape[1] - y_shift[i])

    if overlap_w <= 0:
        # No overlap => copy the whole image directly
        for x in range(r_i):
            for y in range(c_i):
                canvas[x, y + offset] = pics[i][x, y]
        print("  -> No overlap. Image copied directly.\n")
        return None, 0  # No choice map needed

    # 1) Compute pixel-wise difference
    diff = np.zeros((r_i, overlap_w), dtype=np.float32)
    for xx in range(r_i):
        for yy in range(overlap_w):
            px_past = pics[i - 1][xx, yy + offset - y_shift[i - 1]]
            px_now = pics[i][xx, yy]
            val = max(abs(px_past[0] - px_now[0]),
                      abs(px_past[1] - px_now[1]),
                      abs(px_past[2] - px_now[2]))
            diff[xx, yy] = val

    # 2) Compute graph cut to determine blending boundary
    choice = compute_graph_cut(diff, r_i, overlap_w)

    # 3) Copy pixels to the canvas based on graph cut choice
    for xx in range(r_i):
        for yy in range(overlap_w):
            if choice[xx, yy] == 1:
                canvas[xx, yy + offset] = pics[i][xx, yy]
        for yy in range(overlap_w, c_i):
            canvas[xx, yy + offset] = pics[i][xx, yy]

    return choice, overlap_w



def multi_band_blending(imageA, imageB, maskA, maskB):
    h, w = imageA.shape[:2]

    blender = cv2.detail_MultiBandBlender()
    dst_roi = (0, 0, w, h)
    blender.prepare(dst_roi)

    # Convertir a int16
    imgA_16s = imageA.astype(np.int16)
    imgB_16s = imageB.astype(np.int16)

    blender.feed(imgA_16s, maskA, (0, 0))
    blender.feed(imgB_16s, maskB, (0, 0))

    dst = np.zeros((h, w, 3), dtype=np.int16)
    dst_mask = np.zeros((h, w), dtype=np.uint8)

    blender.blend(dst, dst_mask)

    result = np.clip(dst, 0, 255).astype(np.uint8)
    return result

def graphcutNmultiband(dir_in):

    if not dir_in.endswith("/"):
        dir_in += "/"

    metadata_path = os.path.join(dir_in, "metadata.txt")
    if not os.path.exists(metadata_path):
        print(f"{metadata_path} does not exist")
        return

    t0 = time.time()
    pics = []
    cum_yshift = []

    # 1) Read metadata
    with open(metadata_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        fname, shift_str = parts
        shift_val = int(shift_str)
        cum_yshift.append(shift_val)

        img_path = fname  
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read {img_path}")
            return

        pics.append(img.astype(np.float32))

    imgcnt = len(pics)
    print(f"[INFO] Loaded {imgcnt} images from metadata.")

    # 2) Adjust brightness (optional)
    equalize_brightness(pics, cum_yshift)  # if you already have this function
    # 3) Create 'fin' canvas
    h0, w0 = pics[0].shape[:2]
    finalW = int(cum_yshift[imgcnt-1] + w0)
    if finalW < 0:
        finalW = 0
    fin = np.zeros((h0, finalW, 3), dtype=np.float32)
    fin[:h0, :w0] = pics[0]

    mv_cnt = 0
    # Main loop
    for i in range(1, imgcnt):
        print(f"[INFO] Processing image {i}/{imgcnt-1} ...")

        # stitchImageGraphCutOnly => returns (choice, overlap_w)
        choice, overlap_w = stitchImageGraphCutOnly(i, pics, fin, cum_yshift)
        if choice is None:
            # no overlap => skip
            continue
        
        # multi-band blending with cv2.detail_MultiBandBlender
        # 1) Extract subA, subB in the region (r_i, overlap_w)
        r_i, c_i = pics[i].shape[:2]
        offset = cum_yshift[i]

        subA = np.zeros((r_i, overlap_w, 3), dtype=np.float32)
        subB = np.zeros((r_i, overlap_w, 3), dtype=np.float32)
        maskA = np.zeros((r_i, overlap_w), dtype=np.uint8)
        maskB = np.zeros((r_i, overlap_w), dtype=np.uint8)

        for xx in range(r_i):
            for yy in range(overlap_w):
                old_px = fin[xx, yy+offset]   # the "old" part already in fin
                new_px = pics[i][xx, yy]      # the image i
                subA[xx, yy] = old_px
                subB[xx, yy] = new_px

                if choice[xx, yy] == -1:
                    # prefers old => maskA=255, maskB=0
                    maskA[xx, yy] = 255
                    maskB[xx, yy] = 0
                else:
                    # prefers new => maskB=255, maskA=0
                    maskA[xx, yy] = 0
                    maskB[xx, yy] = 255

        # Convert to 8u
        subA_8u = np.clip(subA, 0,255).astype(np.uint8)
        subB_8u = np.clip(subB, 0,255).astype(np.uint8)

        # Call multi_band_blending
        blended_sub = multi_band_blending(subA_8u, subB_8u, maskA, maskB)

        # Copy blended_sub to 'fin'
        for xx in range(r_i):
            for yy in range(overlap_w):
                fin[xx, yy+offset] = blended_sub[xx, yy]


        # Save partial panorama
        out_ = np.clip(fin, 0,255).astype(np.uint8)
        foutname = f"{dir_in}panorama_multiband_{mv_cnt}.jpg"
        cv2.imwrite(foutname, out_)
        mv_cnt += 1
        print(f"    [MultiBand] saved {foutname}")
    # At the end, save
    final_pano = np.clip(fin, 0,255).astype(np.uint8)
    panorama_final_path = os.path.join(dir_in, "panorama_multiband_final.jpg")
    cv2.imwrite(panorama_final_path, final_pano)

    t1 = time.time()
    print(f"[INFO] Completed. MultiBand total={t1 - t0:.2f} sec. Saved {panorama_final_path}")
    return final_pano


import os
import cv2
import time
import numpy as np
from math import fabs
from numba import njit

@njit
def poisson_one_iteration_numba(
    r_i, c_i,
    overlap_w,
    choice,
    dx4, dy4,
    search_p, res, fin, pics_i,
    cum_yshift_i
):

    # 1) Build "vec"
    vec = np.zeros((r_i, c_i, 3), dtype=np.float32)
    for x in range(1, r_i-1):
        for y in range(0, c_i-1):
            if (y < overlap_w) and (choice[x, y] == -1):
                continue

            accum_ = np.zeros(3, dtype=np.float32)
            for k in range(4):
                nx = x + dx4[k]
                ny = y + dy4[k]
                if nx < 1 or nx >= (r_i-1) or ny < 0 or ny >= (c_i-1):
                    continue
                if (y < overlap_w) and (choice[x, y] == -1):
                    continue

                accum_ += (search_p[x, y] - search_p[nx, ny])
            vec[x, y] = accum_

    # 2) alpha1, alpha2
    alpha1 = 0.0
    alpha2 = 0.0
    for x in range(1, r_i-1):
        for y in range(0, c_i-1):
            if (y < overlap_w) and (choice[x, y] == -1):
                continue

            rr = res[x, y]
            alpha1 += rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]

            sp = search_p[x, y]
            vv = vec[x, y]
            alpha2 += (sp[0]*vv[0] + sp[1]*vv[1] + sp[2]*vv[2])

    if abs(alpha2) < 1e-12:
        return (alpha2, 0.0)  # signal to break

    alpha = alpha1 / alpha2

    # 3) Update fin, res
    for x in range(1, r_i-1):
        for y in range(0, c_i-1):
            if (y < overlap_w) and (choice[x, y] == -1):
                continue

            fin[x, y + cum_yshift_i, 0] += alpha * search_p[x, y, 0]
            fin[x, y + cum_yshift_i, 1] += alpha * search_p[x, y, 1]
            fin[x, y + cum_yshift_i, 2] += alpha * search_p[x, y, 2]

            res[x, y, 0] -= alpha * vec[x, y, 0]
            res[x, y, 1] -= alpha * vec[x, y, 1]
            res[x, y, 2] -= alpha * vec[x, y, 2]

    # 4) Calculate gamma1
    gamma1 = 0.0
    for x in range(1, r_i-1):
        for y in range(0, c_i-1):
            if (y < overlap_w) and (choice[x, y] == -1):
                continue
            rr = res[x, y]
            gamma1 += (rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])

    gamma = gamma1 / alpha1

    # 5) search_p = res + gamma * search_p
    for x in range(1, r_i-1):
        for y in range(0, c_i-1):
            if (y < overlap_w) and (choice[x, y] == -1):
                continue
            rr = res[x, y]
            spxy = search_p[x, y]
            search_p[x, y, 0] = rr[0] + gamma * spxy[0]
            search_p[x, y, 1] = rr[1] + gamma * spxy[1]
            search_p[x, y, 2] = rr[2] + gamma * spxy[2]

    return (alpha2, gamma1)

def poisson_blending(choice, overlap_w, pics_i, fin, offset, dx4, dy4, dir_in, mv_cnt):
    """
    Performs Poisson blending (or alpha, etc.)
    in the region where 'choice' is defined.
    """
    print("  -> Starting Poisson blending (Numba).")
    r_i, c_i = pics_i.shape[:2]
    pb_start = time.time()

    res = np.zeros((r_i, c_i, 3), dtype=np.float32)
    for xx in range(1, r_i - 1):
        for yy in range(0, c_i - 1):
            if (yy < overlap_w) and (choice[xx, yy] == -1):
                continue
            sum_ = np.zeros(3, dtype=np.float32)
            for k_ in range(4):
                nx = xx + dx4[k_]
                ny = yy + dy4[k_]
                if nx < 0 or nx >= r_i or ny < 0 or ny >= c_i:
                    continue
                sum_ += (pics_i[xx, yy] - pics_i[nx, ny])
                if 0 <= ny < c_i:
                    fx_ = fin[nx, ny + offset]
                    fx = fin[xx, yy + offset]
                    sum_ += (fx_ - fx)
            res[xx, yy] = sum_

    search_p = res.copy()
    T = 10000
    final_out = None
    dx4_ = np.array(dx4, dtype=np.int32)
    dy4_ = np.array(dy4, dtype=np.int32)

    for t_ in range(T):
        alpha2, gamma1 = poisson_one_iteration_numba(
            r_i, c_i,
            overlap_w,
            choice,
            dx4_, dy4_,
            search_p, res, fin, pics_i,
            offset
        )
        if abs(alpha2) < 1e-12:
            break
        if gamma1 < 200:
            break

        if t_ % 100 == 0:
            out_ = np.clip(fin, 0, 255).astype(np.uint8)
            foutname = f"{dir_in}panorama_{mv_cnt}.jpg"
            cv2.imwrite(foutname, out_)
            mv_cnt += 1
            final_out = out_.copy()
            print(f"    [Poisson] Iter {t_}: Error={gamma1:.2f}, saved {foutname}")

    if final_out is None:
        final_out = np.clip(fin, 0, 255).astype(np.uint8)

    panorama_final_path = os.path.join(dir_in, f"panorama_final_img{mv_cnt}.jpg")
    cv2.imwrite(panorama_final_path, final_out)

    pb_end = time.time()
    print(f"  -> Poisson blending completed in ~iter={t_}, gamma1={gamma1:.2f}, "
          f"time={pb_end - pb_start:.2f}s. Saved {panorama_final_path}\n")

    return mv_cnt, final_out

def graphcutNpoisson(dir_in):
    if not dir_in.endswith("/"):
        dir_in += "/"

    metadata_path = os.path.join(dir_in, "metadata.txt")
    if not os.path.exists(metadata_path):
        print(f"{metadata_path} does not exist")
        return

    t0 = time.time()
    pics = []
    cum_yshift = []

    # Read metadata
    with open(metadata_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        fname, shift_str = parts
        shift_val = int(shift_str)
        cum_yshift.append(shift_val)

        img_path = fname  # or os.path.join(dir_in, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read image {img_path}")
            return

        pics.append(img.astype(np.float32))

    imgcnt = len(pics)
    print(f"[INFO] Loaded {imgcnt} images from metadata.")

    # 2) Adjust brightness
    equalize_brightness(pics, cum_yshift)  # if you have this function

    # 3) Create canvas
    h0, w0 = pics[0].shape[:2]
    finalW = int(cum_yshift[imgcnt - 1] + w0)
    if finalW < 0:
        finalW = 0
    fin = np.zeros((h0, finalW, 3), dtype=np.float32)
    fin[:h0, :w0] = pics[0]

    dx4 = [0, 0, 1, -1]
    dy4 = [1, -1, 0, 0]

    mv_cnt = 0
    for i in range(1, imgcnt):
        print(f"[INFO] Processing image {i}/{imgcnt - 1} ...")

        # 1) Graph cut only
        choice_i, overlap_w = stitchImageGraphCutOnly(i, pics, fin, cum_yshift)
        if choice_i is None:
            # No overlap => skip to the next
            continue

        # 2) Poisson blending
        mv_cnt, final_out = poisson_blending(
            choice_i, overlap_w,
            pics[i], fin,
            cum_yshift[i],
            dx4, dy4,
            dir_in, mv_cnt
        )

    t1 = time.time()
    print(f"[INFO] Completed. Total time={t1 - t0:.2f} seconds.")
    return final_out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panorama stitching using multiband blending and graph cut.")
    parser.add_argument("--input_dir", type=str, default="./pano_images", help="Directory containing source images and img_list.txt")
    parser.add_argument("--processing_dir", type=str, default="temp", help="Directory to store intermediate processed images")
    parser.add_argument("--focal", type=float, default=1700.0, help="Focal length to use for cylindrical warping (default: 1700)")
    parser.add_argument("--blender", type=str, choices=["p", "mb"], default="mb", help="Blending method: 'p' for Poisson, 'mb' for MultiBand (default: mb)")
    args = parser.parse_args()

    print("╭────────────────────────────────────────────╮")
    print("│       📷 Panorama Stitching Pipeline       │")
    print("╰────────────────────────────────────────────╯")
    print(f"[INFO] Input directory       : {args.input_dir}")
    print(f"[INFO] Processing directory  : {args.processing_dir}")
    print(f"[INFO] Focal length (passed) : {args.focal}")
    print(f"[INFO] Blender method        : {args.blender}")
    print("")

    print("[STEP 1] Preprocessing and warping images...")
    stitch_preprocess(args.input_dir, args.processing_dir, focal_length=args.focal)

    if args.blender == "mb":
        print("[STEP 2] Stitching with GraphCut and MultiBand blending...")
        final_pano = graphcutNmultiband(args.processing_dir)
        final_output_path = "./panorama_multiband_final.jpg"
    elif args.blender == "p":
        print("[STEP 2] Stitching with GraphCut and Poisson blending...")
        final_pano = graphcutNpoisson(args.processing_dir)
        final_output_path = "./panorama_poisson_final.jpg"

    if final_pano is not None:
        final_pano_rgb = cv2.cvtColor(final_pano.astype('uint8'), cv2.COLOR_BGR2RGB)
        cv2.imwrite(final_output_path, final_pano)
        print(f"[✅ SUCCESS] Final panorama saved as: {final_output_path}")

        plt.imshow(final_pano_rgb)
        plt.axis("off")
        plt.title("Final Panorama")
        plt.show()
    else:
        print("[❌ ERROR] Final panorama could not be generated.")
