from pathlib import Path


def sammenlign_resultater(orb_data, sift_data):
    orb_vis, orb_matches, orb_kp1, orb_kp2, orb_tid = orb_data
    sift_vis, sift_matches, sift_kp1, sift_kp2, sift_tid = sift_data

    print(f"\nORB+BF:")
    print(f"  Nøkkelpunkter: {orb_kp1} vs {orb_kp2}")
    print(f"  Matches: {len(orb_matches)}")
    print(f"  Tid: {orb_tid:.4f}s")

    print(f"\nSIFT+FLANN:")
    print(f"  Nøkkelpunkter: {sift_kp1} vs {sift_kp2}")
    print(f"  Matches: {len(sift_matches)}")
    print(f"  Tid: {sift_tid:.4f}s")

    # Sammenligning
    if orb_tid < sift_tid:
        prosent = ((sift_tid - orb_tid) / sift_tid * 100)
        print(f"\nORB+BF er {prosent:.1f}% raskere")

    return {
        'orb': {'matches': len(orb_matches), 'kp1': orb_kp1, 'kp2': orb_kp2, 'tid': orb_tid, 'bilde': orb_vis},
        'sift': {'matches': len(sift_matches), 'kp1': sift_kp1, 'kp2': sift_kp2, 'tid': sift_tid, 'bilde': sift_vis}
    }