package dev.aiutopia.mod.mixin;

import dev.aiutopia.mod.agent.AgentRegistry;
import net.minecraft.server.command.KickCommand;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.Collection;

/** Block `/kick` of registered AI agents — for ALL permission levels.
 *
 *  Permission semantics: vanilla `/kick` already requires permission
 *  level 3, so any source that reaches this mixin is already op-3+.
 *  Filtering by permission level here would be a no-op. Instead we
 *  block ALL `/kick` of agents (including from full operators) and
 *  force the explicit `aiutopia agent kill <uuid>` CLI path, so
 *  accidental misclick kicks can never trigger permadeath.
 *
 *  Target verified empirically against MC 1.21.1 Yarn 1.21.1+build.3 via
 *  `javap -p -c minecraft-unpicked.jar KickCommand`:
 *    private static int execute(
 *        ServerCommandSource source,
 *        Collection<ServerPlayerEntity> targets,
 *        Text reason
 *    ) throws CommandSyntaxException;
 *
 *  An earlier code-review note claimed the method was `kick(Collection<GameProfile>, Text)`,
 *  which was incorrect for this Yarn revision — the original `execute(Collection<ServerPlayerEntity>, Text)`
 *  signature is what's actually present.
 */
@Mixin(KickCommand.class)
public abstract class KickPlayerMixin {

    @Inject(method = "execute", at = @At("HEAD"), cancellable = true)
    private static void aiutopia$blockKickOfAgents(
            ServerCommandSource source,
            Collection<ServerPlayerEntity> targets,
            Text reason,
            CallbackInfoReturnable<Integer> cir) {
        for (ServerPlayerEntity target : targets) {
            if (AgentRegistry.isAgent(target.getGameProfile().getName())) {
                source.sendError(Text.literal(
                    "AI Utopia: cannot kick agent '" + target.getGameProfile().getName()
                    + "' via /kick. Use `aiutopia agent kill <uuid>` instead."));
                cir.setReturnValue(0);
                return;
            }
        }
    }
}
