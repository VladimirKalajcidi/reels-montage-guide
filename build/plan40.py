"""Генерирует монтажный лист ролика 8 из storyboard40."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard40 import SHOTS, CAPS, DUR

NAMES = {
    "A1": "R1",
    "num": "R4+R5",
    "stock": "R2",
}

STOCK_DESC = {
    "4705_warehouse_inventory": "работник делает инвентаризацию на складе (Mixkit 4705)",
    "6102_grocery_shelves": "товары на полках продуктового магазина (Mixkit 6102)",
    "22676_browsing_small_grocery": "покупатели в проходе продуктового магазина (Mixkit 22676)",
    "31346_courier_boxes_truck": "курьер грузит коробки в машину доставки (Mixkit 31346)",
    "21818_store_fridge_door": "открывают дверь холодильника в магазине (Mixkit 21818)",
    "2846_rainy_window": "дождь за окном/витриной (Mixkit 2846)",
    "22545_browsing_frozen_foods": "выбор в отделе заморозки (Mixkit 22545)",
    "49137_discounts_supermarket": "покупательница ищет скидки в супермаркете (Mixkit 49137)",
    "46530_christmas_window_shopping": "новогодняя витрина, window-shopping (Mixkit 46530)",
    "45848_forklift_warehouse": "погрузчик на складе (Mixkit 45848)",
    "5660_job_interview_office": "собеседование в офисе (Mixkit 5660)",
    "24890_choosing_groceries": "выбор продуктов в магазине (Mixkit 24890)",
    "5437_hud_graph_animation": "анимированная HUD-графика с данными/трендами (Mixkit 5437)",
    "5238_dice_throw": "бросок игральных костей (Mixkit 5238)",
    "34170_student_homework_desk": "подросток делает уроки за столом (Mixkit 34170)",
}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k == "A1")
lens = [t1 - t0 for t0, t1, _, _ in SHOTS]
lens_sorted = sorted(lens)

print("# Монтажный лист — «аналитик спроса» (ролик 8)\n")
print(f"Исходник: `videos/8/source.mov` · 720×1280 (масштаб под холст 1080×1920) · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `videos/8/analyst_edit.mp4` · 1080×1920 · 30 fps\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens_sorted[len(lens_sorted)//2]:.2f}с · мин {lens_sorted[0]:.2f}с · макс {lens_sorted[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec = NAMES[kind]
    if kind == "stock":
        desc = STOCK_DESC.get(prm["clip"], prm["clip"])
    elif kind == "num":
        desc = f"сетка — число «{prm['digits']}» / «{prm['label']}»" + (" (синее)" if prm.get("blue") else " (белое)")
    else:
        desc = "лицо"
        if prm.get("topword"):
            desc += f" + крупное слово «{prm['topword']}» над головой (R6)"
    txt = "" if kind == "num" else " / ".join(
        "".join(r[0] for r in runs).strip() for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

print("\n## Звук")
print("- SFX: low whoosh (0.040 от пика голоса) на каждом резе, impact (0.070) на обоих числах-ревилах.")
print("- Музыка: `audios/song2.mp3`, −19 dB от RMS голоса, петля с кроссфейдом 0.25с, фейд-ин 0.8с / фейд-аут 2.0с.")
print("- Голос не резан, финальный `loudnorm=I=-14:TP=-1.5:LRA=7`; замер факта — см. `qa-report.md`.")

print("\n## Проверка на повтор стока")
clips = [prm["clip"] for _, _, k, prm in SHOTS if k == "stock"]
print(f"Уникальных клипов: {len(set(clips))} из {len(clips)} использований — повторов нет."
      if len(set(clips)) == len(clips) else "ВНИМАНИЕ: есть повтор клипа!")
