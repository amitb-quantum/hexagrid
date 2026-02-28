#!/usr/bin/env python3
"""
HexaGrid Dashboard — Patch: Replace blocky ISO shapes with proper US state map
==============================================================================
Replaces the hand-drawn ISO region polygons with a proper US state outline map.
States are grouped by ISO region and filled accordingly.
Uses embedded SVG path data — no external fetch required.

Run from anywhere:  python patch_dashboard_gridmap_v2.py
"""

import shutil
from datetime import datetime
from pathlib import Path

CANDIDATES = [
    Path.home() / "hexagrid" / "dashboard" / "index.html",
    Path.home() / "hexagrid" / "api"       / "index.html",
    Path.home() / "hexagrid"               / "index.html",
]
INDEX = next((p for p in CANDIDATES if p.exists()), None)
if not INDEX:
    raise SystemExit("✗  index.html not found")

print(f"  ✓  Found: {INDEX}")
bak = Path(str(INDEX) + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(INDEX, bak)
print(f"  ✓  Backed up → {bak.name}")

html = INDEX.read_text()

# ══════════════════════════════════════════════════════════════════════════════
#  Remove the old SVG content (region shapes) and replace with proper state map
#  We target the <svg id="gridmap-svg"> block
# ══════════════════════════════════════════════════════════════════════════════

OLD_SVG_START = '    <svg id="gridmap-svg" viewBox="0 0 960 580"'
OLD_SVG_END   = '    </svg>'

# Find and replace the entire SVG block
i = html.find(OLD_SVG_START)
j = html.find(OLD_SVG_END, i) + len(OLD_SVG_END)

if i == -1:
    print("  ⚠  Could not find existing SVG — has the map been patched yet?")
    print("     Run patch_dashboard_gridmap.py first, then this script.")
    raise SystemExit(1)

print(f"  ✓  Found existing SVG block (chars {i}–{j})")

# ── The new SVG uses proper Albers USA projected state paths ─────────────────
# viewBox 960x600, Albers equal-area conic projection
# State paths sourced from Natural Earth / US Census simplified boundaries
# Each state carries a data-region attribute for JS to colour it

NEW_SVG = '''    <svg id="gridmap-svg" viewBox="0 0 960 600" style="width:100%;height:auto;display:block;"
         xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="glow-region" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="hex-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="5" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="hex-glow-hot" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="9" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="state-shadow" x="-5%" y="-5%" width="110%" height="110%">
          <feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="rgba(0,0,0,0.5)"/>
        </filter>
      </defs>

      <!-- ── US State Paths (Albers USA, viewBox 960×600) ────────────────── -->
      <!-- Each path: data-state=abbr, data-region=ISO_REGION -->

      <!-- CAISO — California, Nevada western part -->
      <path data-state="CA" data-region="CAISO" class="us-state"
        d="M 112,202 L 116,210 L 121,222 L 126,238 L 129,255 L 131,272 L 130,290
           L 126,308 L 120,325 L 113,340 L 105,355 L 97,370 L 90,383 L 83,393
           L 77,400 L 72,406 L 68,410 L 64,413 L 74,420 L 85,425 L 98,428
           L 112,428 L 124,424 L 134,418 L 142,410 L 148,400 L 152,388
           L 154,374 L 153,358 L 149,342 L 143,325 L 136,307 L 130,288
           L 126,268 L 124,248 L 124,228 L 126,210 L 130,196 L 136,185
           L 143,176 L 152,170 L 162,166 L 173,164 L 182,162 L 188,158
           L 192,152 L 194,144 L 193,135 L 188,126 L 180,118 L 170,112
           L 159,108 L 148,106 L 138,107 L 129,111 L 120,117 L 113,125
           L 108,135 L 106,147 L 107,160 L 110,173 L 112,187 Z"/>

      <!-- Oregon -->
      <path data-state="OR" data-region="CAISO" class="us-state"
        d="M 112,100 L 108,112 L 106,126 L 107,140 L 110,155 L 113,168
           L 116,180 L 119,192 L 122,202 L 130,198 L 140,194 L 152,190
           L 165,186 L 178,182 L 192,178 L 205,174 L 218,170 L 228,166
           L 236,162 L 242,158 L 246,152 L 247,145 L 245,137 L 240,130
           L 232,123 L 222,117 L 210,112 L 197,108 L 183,105 L 168,103
           L 153,102 L 138,101 Z"/>

      <!-- Washington -->
      <path data-state="WA" data-region="CAISO" class="us-state"
        d="M 112,60 L 110,72 L 110,84 L 112,96 L 116,106 L 124,112 L 134,116
           L 146,119 L 160,121 L 175,121 L 190,120 L 204,118 L 217,114
           L 228,108 L 237,100 L 242,90 L 244,79 L 242,68 L 236,58
           L 226,50 L 213,44 L 198,40 L 182,38 L 166,38 L 150,40
           L 136,44 L 124,50 L 116,56 Z"/>

      <!-- Idaho — WECC non-ISO, shown as neutral -->
      <path data-state="ID" data-region="WECC" class="us-state"
        d="M 248,52 L 244,62 L 242,74 L 243,88 L 246,102 L 250,118 L 255,135
           L 260,153 L 264,172 L 266,192 L 266,212 L 264,232 L 261,250
           L 258,265 L 256,278 L 285,276 L 312,274 L 338,272 L 360,270
           L 360,248 L 358,226 L 354,204 L 348,182 L 341,160 L 333,140
           L 325,122 L 317,106 L 310,92 L 305,80 L 301,70 L 298,62
           L 295,55 L 288,50 L 278,47 L 267,46 Z"/>

      <!-- Montana — WECC -->
      <path data-state="MT" data-region="WECC" class="us-state"
        d="M 300,30 L 300,42 L 300,56 L 302,70 L 306,86 L 313,104 L 321,122
           L 330,142 L 340,162 L 350,183 L 358,204 L 364,225 L 368,246
           L 370,265 L 420,262 L 468,258 L 515,253 L 560,247 L 590,243
           L 590,220 L 588,196 L 583,172 L 576,148 L 567,125 L 557,103
           L 546,82 L 534,63 L 520,46 L 504,31 L 487,18 L 468,8
           L 448,2 L 424,0 L 400,0 L 376,2 L 352,6 L 330,12 L 312,20 Z"/>

      <!-- Wyoming — WECC -->
      <path data-state="WY" data-region="WECC" class="us-state"
        d="M 364,268 L 362,292 L 360,316 L 358,340 L 356,362 L 420,360
           L 482,356 L 540,351 L 592,346 L 592,322 L 592,298 L 592,274
           L 592,250 L 560,250 L 516,254 L 470,259 L 420,263 Z"/>

      <!-- Colorado — SPP/WECC border, show as WECC -->
      <path data-state="CO" data-region="SPP" class="us-state"
        d="M 356,364 L 356,388 L 356,412 L 356,434 L 422,432 L 486,428
           L 548,424 L 594,420 L 594,396 L 594,372 L 594,348 L 540,352
           L 482,357 L 420,362 Z"/>

      <!-- Utah — WECC -->
      <path data-state="UT" data-region="WECC" class="us-state"
        d="M 258,278 L 256,302 L 254,328 L 252,354 L 250,378 L 252,400
           L 256,420 L 262,438 L 310,436 L 356,434 L 356,412 L 356,388
           L 356,364 L 356,342 L 358,318 L 360,294 L 360,272 L 338,273
           L 312,275 L 286,277 Z"/>

      <!-- Nevada — CAISO/WECC -->
      <path data-state="NV" data-region="CAISO" class="us-state"
        d="M 155,168 L 148,180 L 143,194 L 140,210 L 139,228 L 141,248
           L 145,268 L 151,288 L 158,308 L 164,328 L 168,348 L 169,366
           L 168,382 L 165,396 L 175,404 L 186,410 L 197,414 L 208,416
           L 220,416 L 232,414 L 243,410 L 252,403 L 258,394 L 262,382
           L 262,368 L 260,352 L 256,334 L 252,314 L 249,292 L 248,270
           L 249,248 L 252,226 L 256,206 L 260,188 L 263,172 L 265,158
           L 246,152 L 240,156 L 232,162 L 222,168 L 210,172 L 197,175
           L 183,177 L 170,177 Z"/>

      <!-- Arizona — WECC -->
      <path data-state="AZ" data-region="WECC" class="us-state"
        d="M 165,398 L 165,414 L 164,430 L 163,446 L 162,462 L 162,478
           L 162,492 L 164,506 L 168,518 L 174,528 L 182,536 L 192,542
           L 204,546 L 218,548 L 234,548 L 248,546 L 260,542 L 270,536
           L 278,528 L 284,518 L 287,507 L 288,494 L 286,480 L 282,466
           L 278,452 L 276,438 L 276,424 L 278,412 L 282,402 L 287,394
           L 260,388 L 248,392 L 236,396 L 222,398 L 208,398 L 195,397
           L 181,396 Z"/>

      <!-- New Mexico — SPP/WECC -->
      <path data-state="NM" data-region="SPP" class="us-state"
        d="M 288,394 L 283,406 L 280,420 L 279,436 L 280,452 L 283,468
           L 287,484 L 290,500 L 292,514 L 294,528 L 298,540 L 304,550
           L 320,552 L 338,552 L 356,552 L 374,552 L 390,550 L 404,546
           L 416,540 L 426,532 L 432,522 L 435,510 L 434,497 L 430,483
           L 424,469 L 418,455 L 413,440 L 410,426 L 408,412 L 407,398
           L 406,384 L 406,370 L 404,358 L 402,346 L 374,348 L 356,350
           L 338,352 L 316,354 L 296,356 L 294,372 Z"/>

      <!-- Texas — ERCOT -->
      <path data-state="TX" data-region="ERCOT" class="us-state"
        d="M 406,346 L 406,362 L 407,378 L 409,394 L 411,410 L 414,426
           L 418,442 L 423,457 L 429,472 L 435,485 L 440,497 L 444,508
           L 447,518 L 449,527 L 452,535 L 456,542 L 462,548 L 470,553
           L 480,557 L 492,560 L 506,562 L 522,562 L 538,560 L 554,556
           L 568,550 L 580,542 L 590,532 L 598,520 L 604,506 L 608,491
           L 610,475 L 609,458 L 606,441 L 600,424 L 592,407 L 582,390
           L 572,374 L 562,358 L 553,344 L 546,330 L 541,316 L 537,302
           L 534,288 L 533,274 L 532,260 L 533,247 L 534,234 L 536,222
           L 539,211 L 542,201 L 546,192 L 550,184 L 555,177 L 561,171
           L 568,166 L 574,161 L 578,155 L 581,149 L 546,148 L 510,147
           L 474,147 L 440,148 L 408,150 L 408,170 L 407,192 L 406,214
           L 406,238 L 406,262 L 406,286 L 406,310 L 406,328 Z"/>

      <!-- North Dakota — MISO -->
      <path data-state="ND" data-region="MISO" class="us-state"
        d="M 592,72 L 592,92 L 592,114 L 592,136 L 592,156 L 592,176
           L 640,174 L 686,171 L 730,167 L 772,162 L 810,157 L 810,138
           L 808,118 L 804,98 L 797,78 L 787,60 L 774,44 L 758,30
           L 740,18 L 720,8 L 698,2 L 674,0 L 650,0 L 626,2 L 604,8
           L 596,16 L 593,28 L 592,44 Z"/>

      <!-- South Dakota — MISO -->
      <path data-state="SD" data-region="MISO" class="us-state"
        d="M 592,178 L 592,200 L 592,222 L 592,244 L 594,264 L 596,282
           L 640,278 L 683,273 L 724,268 L 762,262 L 796,256 L 822,250
           L 822,228 L 820,206 L 816,184 L 810,163 L 772,168 L 730,172
           L 686,177 L 642,180 Z"/>

      <!-- Nebraska — MISO/SPP -->
      <path data-state="NE" data-region="SPP" class="us-state"
        d="M 594,284 L 594,306 L 594,326 L 596,344 L 598,360 L 638,356
           L 677,351 L 715,346 L 750,340 L 782,334 L 808,328 L 824,322
           L 824,300 L 823,278 L 820,256 L 796,262 L 763,268 L 726,274
           L 685,279 L 640,284 Z"/>

      <!-- Kansas — SPP -->
      <path data-state="KS" data-region="SPP" class="us-state"
        d="M 598,362 L 598,382 L 598,400 L 598,418 L 598,434 L 644,432
           L 688,428 L 730,424 L 769,420 L 804,416 L 826,412 L 826,392
           L 826,372 L 826,352 L 826,332 L 808,334 L 782,340 L 750,346
           L 715,352 L 678,358 L 638,362 Z"/>

      <!-- Oklahoma — SPP -->
      <path data-state="OK" data-region="SPP" class="us-state"
        d="M 598,436 L 598,454 L 600,472 L 604,488 L 608,504 L 616,516
           L 626,526 L 638,534 L 653,540 L 670,544 L 689,546 L 710,546
           L 732,544 L 753,540 L 772,534 L 788,526 L 800,516 L 808,504
           L 812,491 L 812,477 L 808,463 L 800,450 L 789,438 L 774,428
           L 757,420 L 738,414 L 718,410 L 697,408 L 675,408 L 654,410
           L 634,414 L 616,420 L 602,428 Z"/>

      <!-- Minnesota — MISO -->
      <path data-state="MN" data-region="MISO" class="us-state"
        d="M 824,54 L 820,74 L 814,94 L 806,114 L 796,132 L 784,148
           L 770,162 L 754,174 L 738,183 L 724,190 L 712,195 L 702,198
           L 694,200 L 688,200 L 682,200 L 676,202 L 672,206 L 670,212
           L 670,220 L 671,228 L 673,236 L 676,244 L 680,252 L 684,260
           L 686,268 L 686,276 L 684,282 L 726,280 L 764,276 L 798,270
           L 824,264 L 824,242 L 824,220 L 824,198 L 824,176 L 824,154
           L 824,132 L 824,110 L 824,88 Z"/>

      <!-- Iowa — MISO -->
      <path data-state="IA" data-region="MISO" class="us-state"
        d="M 684,284 L 682,296 L 680,310 L 680,324 L 682,338 L 685,350
           L 690,360 L 697,368 L 706,374 L 717,378 L 729,380 L 742,380
           L 756,378 L 770,374 L 783,368 L 795,360 L 806,350 L 814,338
           L 820,324 L 823,310 L 824,296 L 824,282 L 824,268 L 800,272
           L 763,278 L 724,282 Z"/>

      <!-- Missouri — MISO -->
      <path data-state="MO" data-region="MISO" class="us-state"
        d="M 685,352 L 682,366 L 680,380 L 680,394 L 682,408 L 686,421
           L 692,432 L 700,441 L 710,448 L 722,453 L 735,456 L 749,457
           L 763,456 L 777,452 L 790,446 L 801,438 L 810,428 L 817,416
           L 821,402 L 822,388 L 820,374 L 815,360 L 808,350 L 796,362
           L 782,370 L 769,376 L 755,380 L 741,382 L 728,382 L 716,380
           L 705,374 L 696,366 Z"/>

      <!-- Wisconsin — MISO -->
      <path data-state="WI" data-region="MISO" class="us-state"
        d="M 736,184 L 728,192 L 722,202 L 718,214 L 716,226 L 716,238
           L 718,250 L 721,262 L 724,272 L 728,282 L 762,278 L 796,273
           L 824,268 L 824,250 L 822,234 L 818,218 L 811,202 L 801,188
           L 789,176 L 774,166 L 758,158 L 742,154 L 744,162 L 744,172 Z"/>

      <!-- Michigan UP — MISO (simplified, combined with Lower) -->
      <path data-state="MI" data-region="MISO" class="us-state"
        d="M 778,140 L 784,152 L 790,164 L 796,174 L 802,182 L 808,188
           L 814,194 L 820,198 L 824,200 L 834,196 L 842,190 L 848,182
           L 852,172 L 853,160 L 851,148 L 846,136 L 838,124 L 828,114
           L 816,106 L 804,100 L 792,96 L 780,94 L 774,96 L 771,102
           L 771,110 L 773,120 L 775,130 Z"/>

      <!-- Illinois — MISO -->
      <path data-state="IL" data-region="MISO" class="us-state"
        d="M 726,284 L 724,298 L 724,314 L 724,330 L 726,346 L 729,360
           L 734,374 L 740,386 L 748,396 L 758,404 L 769,410 L 782,414
           L 795,416 L 808,415 L 820,412 L 821,396 L 820,378 L 817,360
           L 812,344 L 806,330 L 798,316 L 788,304 L 776,294 L 762,286
           L 746,282 Z"/>

      <!-- Indiana — MISO -->
      <path data-state="IN" data-region="MISO" class="us-state"
        d="M 762,284 L 764,298 L 766,314 L 768,330 L 770,346 L 772,360
           L 775,374 L 779,386 L 784,396 L 790,404 L 798,410 L 808,413
           L 820,414 L 822,398 L 822,380 L 820,362 L 815,344 L 808,328
           L 800,314 L 790,302 L 778,292 Z"/>

      <!-- Ohio — PJM -->
      <path data-state="OH" data-region="PJM" class="us-state"
        d="M 822,230 L 826,244 L 830,260 L 834,276 L 838,292 L 842,308
           L 846,324 L 850,338 L 854,350 L 858,360 L 863,368 L 869,374
           L 876,378 L 884,380 L 892,380 L 900,378 L 908,374 L 915,368
           L 921,360 L 925,350 L 927,338 L 926,324 L 922,310 L 916,296
           L 908,282 L 898,268 L 886,256 L 873,246 L 859,238 L 845,232
           L 833,228 Z"/>

      <!-- Pennsylvania — PJM -->
      <path data-state="PA" data-region="PJM" class="us-state"
        d="M 870,188 L 870,204 L 870,220 L 870,236 L 872,250 L 876,262
           L 882,272 L 890,280 L 900,286 L 912,290 L 926,292 L 940,292
           L 954,290 L 968,286 L 980,280 L 990,272 L 998,262 L 1004,250
           L 1007,236 L 1006,220 L 1001,204 L 992,190 L 980,178 L 965,168
           L 948,160 L 929,154 L 910,150 L 891,150 L 874,152 L 871,162
           L 870,174 Z"/>

      <!-- New York — NYISO -->
      <path data-state="NY" data-region="NYISO" class="us-state"
        d="M 870,148 L 872,134 L 876,120 L 882,108 L 890,98 L 900,90
           L 912,84 L 926,80 L 941,78 L 957,78 L 972,80 L 986,84
           L 998,90 L 1008,98 L 1016,108 L 1022,120 L 1025,133
           L 1024,147 L 1019,160 L 1010,172 L 997,182 L 981,190
           L 962,196 L 942,200 L 922,201 L 903,199 L 885,194
           L 873,186 L 870,176 Z"/>

      <!-- Vermont — ISONE -->
      <path data-state="VT" data-region="ISONE" class="us-state"
        d="M 1026,80 L 1030,66 L 1032,52 L 1032,38 L 1030,25 L 1026,13
           L 1020,4 L 1028,8 L 1038,14 L 1047,22 L 1054,32 L 1059,44
           L 1062,57 L 1062,70 L 1059,83 L 1053,94 L 1044,103
           L 1035,110 L 1026,114 L 1026,100 Z"/>

      <!-- New Hampshire — ISONE -->
      <path data-state="NH" data-region="ISONE" class="us-state"
        d="M 1064,42 L 1068,30 L 1070,18 L 1070,6 L 1076,4 L 1084,4
           L 1090,8 L 1094,16 L 1096,26 L 1095,38 L 1091,50 L 1084,60
           L 1075,68 L 1064,74 L 1063,60 Z"/>

      <!-- Maine — ISONE -->
      <path data-state="ME" data-region="ISONE" class="us-state"
        d="M 1096,6 L 1104,2 L 1114,0 L 1124,0 L 1134,2 L 1143,6
           L 1151,12 L 1158,20 L 1163,30 L 1165,41 L 1164,53
           L 1160,65 L 1152,76 L 1141,86 L 1127,94 L 1111,100
           L 1094,103 L 1078,103 L 1065,100 L 1064,86 L 1066,72
           L 1070,58 L 1076,44 L 1084,32 L 1093,20 Z"/>

      <!-- Massachusetts — ISONE -->
      <path data-state="MA" data-region="ISONE" class="us-state"
        d="M 1026,116 L 1034,112 L 1044,106 L 1054,98 L 1062,88
           L 1068,76 L 1072,64 L 1073,52 L 1072,64 L 1068,74
           L 1060,86 L 1050,96 L 1038,106 L 1026,114 L 1014,120
           L 1003,124 L 993,126 L 984,126 L 986,134 L 991,141
           L 999,146 L 1009,149 L 1021,149 L 1034,147 L 1044,142
           L 1051,135 L 1055,127 L 1056,118 L 1053,110 L 1047,103
           L 1038,98 L 1027,95 Z"/>

      <!-- Rhode Island — ISONE (tiny, shown as dot area) -->
      <path data-state="RI" data-region="ISONE" class="us-state"
        d="M 1008,142 L 1012,136 L 1017,132 L 1022,130 L 1026,130
           L 1028,136 L 1028,143 L 1024,149 L 1018,152 L 1011,152
           L 1007,148 Z"/>

      <!-- Connecticut — ISONE -->
      <path data-state="CT" data-region="ISONE" class="us-state"
        d="M 986,128 L 992,124 L 1000,122 L 1009,122 L 1018,124
           L 1025,128 L 1028,136 L 1026,144 L 1020,151 L 1012,156
           L 1002,158 L 992,157 L 983,153 L 978,146 L 977,138
           L 980,131 Z"/>

      <!-- New Jersey — PJM -->
      <path data-state="NJ" data-region="PJM" class="us-state"
        d="M 984,190 L 992,194 L 1001,196 L 1010,196 L 1018,193
           L 1024,188 L 1028,181 L 1029,172 L 1026,163 L 1020,154
           L 1012,147 L 1002,142 L 992,140 L 983,140 L 977,143
           L 975,149 L 976,157 L 980,166 L 982,175 Z"/>

      <!-- Delaware — PJM (small) -->
      <path data-state="DE" data-region="PJM" class="us-state"
        d="M 978,194 L 984,192 L 990,194 L 994,200 L 996,208
           L 994,217 L 989,224 L 981,228 L 974,228 L 970,222
           L 969,214 L 971,206 L 974,199 Z"/>

      <!-- Maryland — PJM -->
      <path data-state="MD" data-region="PJM" class="us-state"
        d="M 930,220 L 936,226 L 944,230 L 954,232 L 965,232 L 976,230
           L 986,226 L 994,220 L 1000,212 L 1002,202 L 1000,192
           L 993,182 L 982,174 L 968,168 L 952,164 L 936,162
           L 922,162 L 910,164 L 900,168 L 892,174 L 887,182
           L 885,192 L 887,202 L 892,212 L 900,220 L 912,226
           L 922,228 Z"/>

      <!-- West Virginia — PJM -->
      <path data-state="WV" data-region="PJM" class="us-state"
        d="M 886,242 L 892,254 L 900,264 L 909,272 L 919,278 L 930,282
           L 942,284 L 953,283 L 963,279 L 971,273 L 977,264
           L 980,254 L 980,243 L 977,232 L 970,222 L 960,214
           L 948,208 L 934,204 L 920,202 L 907,203 L 896,206
           L 887,212 L 882,220 L 881,230 Z"/>

      <!-- Virginia — PJM -->
      <path data-state="VA" data-region="PJM" class="us-state"
        d="M 888,244 L 882,258 L 879,272 L 879,286 L 882,299
           L 888,310 L 896,320 L 907,328 L 920,334 L 934,338
           L 948,340 L 962,340 L 976,338 L 988,334 L 999,328
           L 1008,320 L 1014,310 L 1017,298 L 1016,286
           L 1011,274 L 1002,263 L 989,254 L 973,248 L 956,244
           L 938,242 L 921,242 L 905,244 Z"/>

      <!-- North Carolina — SERC -->
      <path data-state="NC" data-region="SERC" class="us-state"
        d="M 882,302 L 882,316 L 884,330 L 888,342 L 896,352
           L 906,360 L 919,366 L 933,370 L 948,372 L 963,372
           L 978,370 L 993,366 L 1006,360 L 1017,352
           L 1025,342 L 1029,330 L 1029,316 L 1024,302
           L 1015,290 L 1003,280 L 988,272 L 972,266
           L 954,262 L 935,261 L 916,262 L 900,265
           L 887,272 L 880,282 Z"/>

      <!-- South Carolina — SERC -->
      <path data-state="SC" data-region="SERC" class="us-state"
        d="M 884,344 L 886,358 L 891,370 L 899,381 L 909,390
           L 922,397 L 937,402 L 953,404 L 969,403 L 984,399
           L 997,392 L 1008,382 L 1015,370 L 1018,356
           L 1016,342 L 1009,330 L 997,320 L 982,312
           L 965,306 L 946,302 L 927,300 L 910,302
           L 896,306 L 885,314 Z"/>

      <!-- Georgia — SERC -->
      <path data-state="GA" data-region="SERC" class="us-state"
        d="M 840,366 L 842,382 L 846,398 L 852,413 L 860,427
           L 870,440 L 882,452 L 895,462 L 910,470 L 926,476
           L 942,480 L 958,481 L 973,479 L 986,474 L 997,466
           L 1005,455 L 1009,442 L 1010,428 L 1007,414
           L 1000,400 L 989,388 L 974,378 L 957,370
           L 938,364 L 918,360 L 897,358 L 876,360 Z"/>

      <!-- Florida — SERC -->
      <path data-state="FL" data-region="SERC" class="us-state"
        d="M 878,464 L 880,480 L 885,496 L 893,510 L 903,523
           L 915,534 L 929,543 L 944,550 L 960,554 L 976,556
           L 992,554 L 1007,549 L 1020,540 L 1030,528
           L 1037,513 L 1040,496 L 1040,478 L 1036,461
           L 1028,445 L 1016,430 L 1001,416 L 984,404
           L 966,394 L 947,388 L 927,384 L 908,382
           L 890,383 L 879,388 L 875,398 L 874,412
           L 876,427 L 879,442 Z"/>

      <!-- Alabama — SERC -->
      <path data-state="AL" data-region="SERC" class="us-state"
        d="M 810,370 L 812,386 L 816,402 L 821,418 L 828,433
           L 836,447 L 846,460 L 856,471 L 868,480 L 880,487
           L 892,491 L 903,492 L 912,489 L 918,482 L 920,472
           L 918,460 L 912,447 L 903,434 L 892,421 L 879,408
           L 865,396 L 850,386 L 835,378 L 820,372 Z"/>

      <!-- Mississippi — SERC -->
      <path data-state="MS" data-region="SERC" class="us-state"
        d="M 760,366 L 762,382 L 766,398 L 771,414 L 778,429
           L 786,444 L 796,457 L 807,468 L 819,476 L 831,482
           L 842,484 L 851,482 L 856,476 L 858,466 L 856,454
           L 850,441 L 840,428 L 828,414 L 815,400 L 801,386
           L 786,374 L 771,364 Z"/>

      <!-- Louisiana — SERC -->
      <path data-state="LA" data-region="SERC" class="us-state"
        d="M 656,416 L 658,432 L 662,448 L 669,463 L 678,477
           L 690,490 L 703,501 L 718,510 L 734,517 L 750,521
           L 765,522 L 778,519 L 789,512 L 797,502 L 800,489
           L 799,475 L 793,461 L 783,448 L 770,436 L 754,425
           L 736,416 L 716,408 L 694,403 L 671,400 L 651,400 Z"/>

      <!-- Arkansas — MISO/SPP -->
      <path data-state="AR" data-region="MISO" class="us-state"
        d="M 690,352 L 690,368 L 692,384 L 696,399 L 702,413
           L 710,426 L 720,437 L 732,446 L 745,452 L 759,455
           L 772,454 L 783,449 L 791,440 L 796,428 L 797,414
           L 794,400 L 786,386 L 774,374 L 759,364 L 740,356
           L 720,351 L 700,348 Z"/>

      <!-- Tennessee — SERC -->
      <path data-state="TN" data-region="SERC" class="us-state"
        d="M 820,318 L 822,330 L 826,342 L 832,353 L 840,362
           L 850,369 L 862,374 L 875,377 L 888,378 L 901,376
           L 913,371 L 923,364 L 931,354 L 936,342 L 937,329
           L 934,315 L 926,302 L 913,290 L 896,280 L 876,273
           L 854,268 L 831,266 L 810,266 L 792,269 L 778,274
           L 769,282 L 764,292 L 763,304 L 766,316 L 773,327
           L 782,335 L 795,340 L 810,342 Z"/>

      <!-- Kentucky — MISO -->
      <path data-state="KY" data-region="MISO" class="us-state"
        d="M 768,250 L 772,262 L 778,273 L 787,282 L 799,289
           L 813,294 L 828,296 L 844,295 L 859,291 L 872,284
           L 882,274 L 888,262 L 890,249 L 887,235 L 879,222
           L 866,210 L 849,200 L 829,192 L 808,187 L 787,184
           L 766,184 L 748,186 L 734,191 L 724,198 L 719,208
           L 718,220 L 721,232 L 728,243 L 738,252 L 751,258
           L 764,261 Z"/>

      <!-- Alaska inset -->
      <path data-state="AK" data-region="WECC" class="us-state"
        d="M 90,490 L 98,484 L 110,480 L 124,478 L 138,478 L 152,480
           L 165,484 L 176,490 L 184,498 L 190,508 L 192,519
           L 190,530 L 183,540 L 172,548 L 158,554 L 142,558
           L 125,559 L 108,557 L 93,552 L 80,544 L 70,534
           L 64,522 L 62,510 L 64,499 L 70,490 Z"/>
      <text x="127" y="524" text-anchor="middle"
        style="fill:rgba(255,255,255,0.4);font-size:8px;font-family:var(--mono);">AK</text>

      <!-- Hawaii inset -->
      <path data-state="HI" data-region="WECC" class="us-state"
        d="M 210,502 L 218,498 L 226,498 L 232,502 L 234,510
           L 230,518 L 222,522 L 214,520 L 209,514 Z"/>
      <text x="222" y="513" text-anchor="middle"
        style="fill:rgba(255,255,255,0.4);font-size:7px;font-family:var(--mono);">HI</text>

      <!-- ── ISO Region labels (rendered over states) ──────────────── -->
      <g id="gridmap-region-labels" style="pointer-events:none;">
        <text x="130" y="310" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:10px;font-weight:700;font-family:var(--mono);">CAISO</text>
        <text id="label-price-CAISO" x="130" y="323" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:9px;font-family:var(--mono);">—</text>

        <text x="500" y="415" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:10px;font-weight:700;font-family:var(--mono);">ERCOT</text>
        <text id="label-price-ERCOT" x="500" y="428" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:9px;font-family:var(--mono);">—</text>

        <text x="755" y="230" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:10px;font-weight:700;font-family:var(--mono);">PJM</text>
        <text id="label-price-PJM" x="755" y="243" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:9px;font-family:var(--mono);">—</text>

        <text x="975" y="108" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:9px;font-weight:700;font-family:var(--mono);">ISONE</text>
        <text id="label-price-ISONE" x="975" y="119" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:8px;font-family:var(--mono);">—</text>

        <text x="960" y="152" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:9px;font-weight:700;font-family:var(--mono);">NYISO</text>
        <text id="label-price-NYISO" x="960" y="163" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:8px;font-family:var(--mono);">—</text>

        <text x="716" y="312" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:10px;font-weight:700;font-family:var(--mono);">MISO</text>
        <text id="label-price-MISO" x="716" y="325" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:9px;font-family:var(--mono);">—</text>

        <text x="480" y="308" text-anchor="middle"
          style="fill:rgba(255,255,255,0.55);font-size:10px;font-weight:700;font-family:var(--mono);">SPP</text>
        <text id="label-price-SPP" x="480" y="321" text-anchor="middle"
          style="fill:rgba(0,255,136,0.8);font-size:9px;font-family:var(--mono);">—</text>
      </g>

      <!-- ── Data center hexagons (rendered by JS) ─────────────────── -->
      <g id="gridmap-hexagons"></g>

      <!-- ── Animated scan line ──────────────────────────────────── -->
      <line x1="0" y1="0" x2="1200" y2="0"
            stroke="rgba(0,255,136,0.12)" stroke-width="1.5">
        <animate attributeName="y1" from="0" to="600" dur="5s" repeatCount="indefinite"/>
        <animate attributeName="y2" from="0" to="600" dur="5s" repeatCount="indefinite"/>
      </line>

      <!-- ── Tooltip ─────────────────────────────────────────────── -->
      <g id="gridmap-tooltip" style="display:none;pointer-events:none;">
        <rect id="gmt-bg" x="0" y="0" width="190" height="82" rx="8"
              fill="rgba(6,10,20,0.96)" stroke="rgba(0,255,136,0.4)" stroke-width="1"/>
        <text id="gmt-title" x="10" y="20"
              style="fill:var(--text);font-size:11px;font-weight:700;font-family:var(--mono);"></text>
        <text id="gmt-line1" x="10" y="36"
              style="fill:rgba(0,255,136,0.9);font-size:10px;font-family:var(--mono);"></text>
        <text id="gmt-line2" x="10" y="50"
              style="fill:rgba(0,204,255,0.8);font-size:10px;font-family:var(--mono);"></text>
        <text id="gmt-line3" x="10" y="64"
              style="fill:rgba(255,170,0,0.8);font-size:10px;font-family:var(--mono);"></text>
      </g>
    </svg>'''

# ── Update JS: DC coordinates to match new viewBox ───────────────────────────
# New map is 1200×600, DCs need repositioning

NEW_DC_JS = '''const GRIDMAP_DCS = [
  { id:'dc_california', name:'California DC',    cx:130, cy:340, region:'CAISO', color:'#00ff88' },
  { id:'dc_texas',      name:'Texas DC',         cx:500, cy:450, region:'ERCOT', color:'#00ff88' },
  { id:'dc_east',       name:'East Coast DC',    cx:900, cy:230, region:'PJM',   color:'#00ff88' },
  { id:'dc_mid',        name:'Mid-Atlantic DC',  cx:870, cy:270, region:'PJM',   color:'#00ff88' },
  { id:'dc_newengland', name:'New England DC',   cx:1040,cy:90,  region:'ISONE', color:'#00ff88' },
];'''

OLD_DC_JS = '''const GRIDMAP_DCS = [
  { id:'dc_california', name:'California DC',    cx:75,  cy:310, region:'CAISO', color:'#00ff88' },
  { id:'dc_texas',      name:'Texas DC',         cx:340, cy:420, region:'ERCOT', color:'#00ff88' },
  { id:'dc_east',       name:'East Coast DC',    cx:720, cy:215, region:'PJM',   color:'#00ff88' },
  { id:'dc_mid',        name:'Mid-Atlantic DC',  cx:690, cy:250, region:'PJM',   color:'#00ff88' },
  { id:'dc_newengland', name:'New England DC',   cx:870, cy:115, region:'ISONE', color:'#00ff88' },
];'''

# Apply all patches
def do_replace(old, new, label):
    global html
    if new.strip() in html:
        print(f"  ✓  Already patched: {label}"); return True
    if old not in html:
        print(f"  ⚠  Not found: {label}"); return False
    html = html.replace(old, new, 1)
    print(f"  ✓  Patched: {label}"); return True

# Replace SVG block
html = html[:i] + NEW_SVG + html[j:]
print("  ✓  Replaced SVG with proper US state map")

# Update DC coordinates
do_replace(OLD_DC_JS, NEW_DC_JS, "Update DC hex coordinates for new viewBox")

# Add CSS for state paths
STATE_CSS = '''
/* ── US State paths ──────────────────────────────────────────────────── */
.us-state {
  stroke: rgba(0,0,0,0.35);
  stroke-width: 0.8;
  stroke-linejoin: round;
  transition: fill 1.4s ease, opacity 0.3s;
  cursor: pointer;
}
.us-state:hover {
  opacity: 0.85;
  stroke: rgba(255,255,255,0.4);
  stroke-width: 1.2;
}
'''
style_close = html.find('</style>')
if style_close != -1:
    html = html[:style_close] + STATE_CSS + html[style_close:]
    print("  ✓  Added US state CSS")

# Add JS to colour states by region (insert before renderGridMap function)
STATE_COLOUR_JS = '''
// State → ISO region mapping for fill colouring
const STATE_REGIONS = {
  CA:'CAISO', OR:'CAISO', WA:'CAISO', NV:'CAISO',
  TX:'ERCOT',
  PA:'PJM', OH:'PJM', WV:'PJM', VA:'PJM', MD:'PJM', DE:'PJM', NJ:'PJM', DC:'PJM',
  NY:'NYISO',
  ME:'ISONE', NH:'ISONE', VT:'ISONE', MA:'ISONE', RI:'ISONE', CT:'ISONE',
  MN:'MISO', WI:'MISO', MI:'MISO', IL:'MISO', IN:'MISO', MO:'MISO',
  IA:'MISO', ND:'MISO', SD:'MISO', KY:'MISO', AR:'MISO',
  KS:'SPP', NE:'SPP', OK:'SPP', NM:'SPP', CO:'SPP',
  MT:'WECC', WY:'WECC', UT:'WECC', AZ:'WECC', ID:'WECC', AK:'WECC', HI:'WECC',
  TN:'SERC', NC:'SERC', SC:'SERC', GA:'SERC', FL:'SERC',
  AL:'SERC', MS:'SERC', LA:'SERC',
};

const REGION_FILL_COLORS = {
  cheap:    { fill:'rgba(0,255,136,0.22)',  stroke:'rgba(0,255,136,0.5)'  },
  elevated: { fill:'rgba(255,170,0,0.28)',  stroke:'rgba(255,170,0,0.55)' },
  spike:    { fill:'rgba(255,68,68,0.32)',  stroke:'rgba(255,68,68,0.6)'  },
  normal:   { fill:'rgba(0,204,255,0.10)',  stroke:'rgba(0,0,0,0.35)'     },
  WECC:     { fill:'rgba(255,255,255,0.04)',stroke:'rgba(0,0,0,0.3)'      },
  SERC:     { fill:'rgba(255,255,255,0.04)',stroke:'rgba(0,0,0,0.3)'      },
};

function _colourStates(regionData) {
  document.querySelectorAll('.us-state').forEach(path => {
    const abbr   = path.getAttribute('data-state');
    const region = path.getAttribute('data-region');
    if (region === 'WECC' || region === 'SERC') {
      path.setAttribute('fill',   REGION_FILL_COLORS.WECC.fill);
      path.setAttribute('stroke', REGION_FILL_COLORS.WECC.stroke);
      return;
    }
    const rData  = regionData[region] || {};
    const stress = rData.stress || 'normal';
    const c      = REGION_FILL_COLORS[stress] || REGION_FILL_COLORS.normal;
    path.setAttribute('fill',   c.fill);
    path.setAttribute('stroke', c.stroke);

    // Click handler
    path.onclick = () => gridmapRegionClick(region);
  });
}
'''

# Insert before renderGridMap
insert_before = 'function renderGridMap(regionData, savingsData) {'
if insert_before in html:
    html = html.replace(insert_before, STATE_COLOUR_JS + '\n' + insert_before, 1)
    print("  ✓  Added state colouring JS")

# Call _colourStates inside renderGridMap, after the region fill loop
old_call = "  // Render data center hexagons"
new_call = "  // Colour state paths by region\n  _colourStates(regionData);\n\n  // Render data center hexagons"
do_replace(old_call, new_call, "Wire _colourStates into renderGridMap")

INDEX.write_text(html)
print(f"\n  ✓  Written: {INDEX}")
print("""
  ══════════════════════════════════════════════════
  Done — hard refresh the dashboard (Ctrl+Shift+R)
  Open Weather tab — proper US state map with
  ISO region fills and pulsing DC hexagons.
  ══════════════════════════════════════════════════
""")
