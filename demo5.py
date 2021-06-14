# -*- codding = utf8 -*-
# @Time : 2020/12/12 14:34
# @Author : SupercarryJason
# @File : demo5.py
# @Software : PyCharm
print("------ 商品列表 ------")
products = [["iphone", 6888], ["MacPro", 14800], ["小米6", 2499], ["Coffee", 31], ["Book", 60], ["Nike", 699]]
for product in products:
    print(products.index(product), end="\t")
    for k in product:
        print(k, end="\t")
    print("")

a = input("请输入你想购买的商品编号，若已完成请输入finished： ")
cart = []
price = 0
while a != "finished":
    cart.append(products[int(a)])
    a = input("请输入你想购买的商品编号，若已完成请输入finished： ")

print("您的购物车为： ")
print(cart)
print("总共的价格为： ")
for items in cart:
    price = price + items[1]
print(price)

