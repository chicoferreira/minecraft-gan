package dev.chicoferreira.rtp;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.jetbrains.annotations.NotNull;

import java.util.Random;

public final class Main extends JavaPlugin implements CommandExecutor {

    private static final Random RANDOM = new Random();

    @Override
    public void onEnable() {
        getCommand("rtp").setExecutor(this);
    }

    public int generateRandomInt(int min, int max) {
        return RANDOM.nextInt((max - min) + 1) + min;
    }

    static int MAX_Y_COORDINATE = 256;

    public Location generateRandomLocation(World world, int coordinateLimit, float maxYOffset) {
        int randomX = generateRandomInt(-coordinateLimit, coordinateLimit);
        int randomZ = generateRandomInt(-coordinateLimit, coordinateLimit);

        int maxY = MAX_Y_COORDINATE;

        for (int i = MAX_Y_COORDINATE - 1; i >= 0; i--) {
            Block block = world.getBlockAt(randomX, i, randomZ);
            if (block.getType() != Material.AIR) {
                maxY = i;
                break;
            }
        }

        float offsetY = RANDOM.nextFloat() * maxYOffset;

        float randomYaw = RANDOM.nextFloat(360);
        float randomPitch = RANDOM.nextFloat(-10.0F, 40.0F);

        return new Location(world, randomX, maxY + offsetY, randomZ, randomYaw, randomPitch);
    }

    @Override
    public boolean onCommand(@NotNull CommandSender sender, @NotNull Command command, @NotNull String label, String @NotNull [] args) {
        if (!(sender instanceof Player player)) {
            return false;
        }

        int coordinateLimit = 5000;
        if (args.length != 0) {
            coordinateLimit = Integer.parseInt(args[0]);
        }

        Location location;
        do {
            location = generateRandomLocation(player.getWorld(), coordinateLimit, 10);
        } while (location.getBlock().getBiome().getKey().getKey().toUpperCase().contains("OCEAN"));

        player.teleport(location);

        return true;
    }
}
