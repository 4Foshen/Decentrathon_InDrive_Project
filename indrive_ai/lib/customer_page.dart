import 'dart:math';
import 'package:flutter/material.dart';
import 'package:indrive_ai/driver/driver_storage.dart';
import 'package:indrive_ai/theme/app_colors.dart';

class CustomerPage extends StatefulWidget {
  const CustomerPage({super.key});

  @override
  State<CustomerPage> createState() => _CustomerPageState();
}

class _CustomerPageState extends State<CustomerPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _animation;

  final List<Map<String, dynamic>> orders = [];

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800));
    _animation =
        Tween<Offset>(begin: const Offset(0, 1), end: Offset.zero).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOut),
    );

    _loadOrders();
    _controller.forward();
  }

  Future<void> _loadOrders() async {
    final savedDrivers = await DriverStorage.loadDrivers();
    setState(() {
      orders.addAll(
          savedDrivers.map((driver) => _generateOrder(driver)).toList());
    });
  }

  Map<String, dynamic> _generateOrder(Map<String, dynamic> driver) {
    final random = Random();

    // Рандомная цена от 500 до 1500, шаг 100
    int price = (random.nextInt(11) + 5) * 100;

    // Рандомная аватарка
    bool isMale = random.nextBool();
    int avatarId = random.nextInt(90); // от 0 до 89
    String avatarUrl =
        "https://randomuser.me/api/portraits/${isMale ? "men" : "women"}/$avatarId.jpg";

    return {
      "name": driver["name"],
      "car": driver["car"],
      "cleanliness": driver["cleanliness"],
      "integrity": driver["integrity"],
      "price": "$price₸",
      "photo": avatarUrl,
    };
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Widget _buildStatusContainer(String status) {
    Color bgColor;
    switch (status) {
      case "битая" || "грязная":
        bgColor = Colors.red.shade600;
        break;
      case "чистая" || "не битая":
        bgColor = Colors.green.shade600;
        break;
      default:
        bgColor = Colors.grey.shade400;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        status,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildOrderCard(Map<String, dynamic> order) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Material(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        elevation: 5,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              CircleAvatar(
                  radius: 30, backgroundImage: NetworkImage(order["photo"])),
              const SizedBox(width: 12),

              // Имя + машина + статус
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(order["name"],
                        style: const TextStyle(
                            color: AppColors.textPrimary,
                            fontSize: 16,
                            fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    Text(order["car"],
                        style: const TextStyle(
                            color: AppColors.textSecondary, fontSize: 14)),
                    const SizedBox(height: 8),

                    // 👇 Две метки
                    Row(
                      children: [
                        _buildStatusContainer(order["cleanliness"]),
                        const SizedBox(width: 6),
                        _buildStatusContainer(order["integrity"]),
                      ],
                    ),
                  ],
                ),
              ),

              // Цена + кнопка
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(order["price"],
                      style: const TextStyle(
                          color: AppColors.primary,
                          fontSize: 16,
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10)),
                    ),
                    onPressed: () {
                      _acceptOrder(order);
                    },
                    child: const Text("Принять",
                        style: TextStyle(
                            color: AppColors.textPrimary, fontSize: 14)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _acceptOrder(Map<String, dynamic> order) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => Dialog(
        backgroundColor: AppColors.background,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.check_circle, color: Colors.green, size: 64),
              const SizedBox(height: 16),
              const Text(
                "Заказ принят!",
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary),
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  Navigator.of(context).pop(); // закрыть диалог
                  setState(() {
                    orders.remove(order);
                  });
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade600,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 40, vertical: 12),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16)),
                  elevation: 4,
                ),
                child: const Text(
                  "Ок",
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: buildTransparentAppBar(context, title: "Заявки"),
      backgroundColor: AppColors.background,
      body: Stack(
        children: [
          // Фон
          Positioned.fill(
            child: Image.asset("assets/img/bg.png", fit: BoxFit.cover),
          ),
          Container(color: Colors.black.withOpacity(0.2)),

          // Список заявок с анимацией
          SlideTransition(
            position: _animation,
            child: Align(
              alignment: Alignment.bottomCenter,
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  // Если заказов ≤ 3 → подстраиваем высоту по их количеству
                  // Если > 3 → ограничиваем высоту и делаем скролл
                  maxHeight: orders.length <= 3
                      ? (orders.length * 110).toDouble() +
                          30 // примерная высота
                      : MediaQuery.of(context).size.height * 0.45,
                ),
                child: orders.length <= 3
                    ? Column(
                        mainAxisSize: MainAxisSize.min,
                        children: orders
                            .map((order) => _buildOrderCard(order))
                            .toList(),
                      )
                    : ListView.builder(
                        itemCount: orders.length,
                        itemBuilder: (context, index) {
                          return _buildOrderCard(orders[index]);
                        },
                      ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

AppBar buildTransparentAppBar(BuildContext context, {String? title}) {
  return AppBar(
    backgroundColor: Colors.transparent,
    elevation: 0,
    leading: IconButton(
      icon: const Icon(Icons.arrow_back, color: Colors.white),
      onPressed: () => Navigator.pop(context),
    ),
    title: title != null
        ? Text(title, style: const TextStyle(color: Colors.white, fontSize: 18))
        : null,
    centerTitle: true,
  );
}
