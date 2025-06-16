import time
import os
import pygetwindow as gw
import pyautogui


class MinecraftBot:
    def __init__(self, screenshot_folder="screenshots", run_until=None):
        self.screenshot_folder = screenshot_folder
        self.run_until = run_until
        self.minecraft_window = None
        self.setup_folders()

    def setup_folders(self):
        os.makedirs(f"{self.screenshot_folder}/before", exist_ok=True)
        os.makedirs(f"{self.screenshot_folder}/after", exist_ok=True)

    def find_minecraft_window(self):
        try:
            windows = gw.getAllWindows()
            for window in windows:
                if "Minecraft* 1.21.5" in window.title:
                    self.minecraft_window = window
                    return window
            return None
        except Exception:
            return None

    def focus_minecraft_window(self):
        if self.minecraft_window:
            try:
                self.minecraft_window.activate()
                time.sleep(0.1)
                return True
            except Exception:
                return False
        return False

    def get_next_id(self):
        """Get the next available ID by finding the highest existing ID"""
        max_id = 0

        # Check both before and after folders
        for folder in ["before", "after"]:
            folder_path = f"{self.screenshot_folder}/{folder}"
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".png"):
                        try:
                            # Extract ID from filename (e.g., "before_123.png" -> 123)
                            id_part = filename.split("_")[-1].split(".")[0]
                            file_id = int(id_part)
                            max_id = max(max_id, file_id)
                        except (ValueError, IndexError):
                            continue

        return max_id + 1

    def take_screenshot(self, filename):
        if not self.minecraft_window:
            return None

        try:
            # Offset to exclude window borders and title bar
            border_offset = 16  # pixels for window border
            title_bar_height = 32  # pixels for title bar

            left = self.minecraft_window.left + border_offset
            top = self.minecraft_window.top + title_bar_height
            width = self.minecraft_window.width - (border_offset * 2)
            height = self.minecraft_window.height - title_bar_height - border_offset

            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot.save(filename)
            return filename

        except Exception:
            return None

    def send_command(self, command):
        if not self.focus_minecraft_window():
            return False

        try:
            pyautogui.press("t")
            time.sleep(0.1)
            pyautogui.write(command)
            time.sleep(0.1)
            pyautogui.press("enter")
            return True

        except Exception:
            return False

    def press_key(self, key):
        if not self.focus_minecraft_window():
            return False

        try:
            pyautogui.press(key)
            return True
        except Exception:
            return False

    def run_collection_loop(self):
        if not self.find_minecraft_window():
            return

        try:
            while True:
                current_id = self.get_next_id()

                if self.run_until and current_id > self.run_until:
                    print(f"Reached {self.run_until}. Stopping collection.")
                    break

                before_filename = f"{self.screenshot_folder}/before/before_{current_id:04d}.png"
                after_filename = f"{self.screenshot_folder}/after/after_{current_id:04d}.png"

                print(f"Taking screenshot {current_id}")

                print(f"{current_id}: /rtp 4800")
                self.send_command("/rtp 4800")
                time.sleep(3)
                print(f"{current_id}: Taking screenshot before")
                self.take_screenshot(before_filename)
                print(f"{current_id}: Pressing K")
                self.press_key("k")
                time.sleep(3)
                print(f"{current_id}: Taking screenshot after")
                self.take_screenshot(after_filename)
                print(f"{current_id}: Pressing K")
                self.press_key("k")
                time.sleep(0.3)

        except KeyboardInterrupt:
            pass


def main():
    time.sleep(5)
    bot = MinecraftBot(run_until=3600)
    bot.run_collection_loop()


if __name__ == "__main__":
    main()
